"""Train a compact character LM using a fixed connectome recurrent core."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from graph import build_graph
from model import ConnectomeLM


def batches(data, batch_size, context, steps, rng):
    for _ in range(steps):
        starts = rng.integers(0, len(data)-context-1, batch_size)
        x = np.stack([data[s:s+context] for s in starts]).astype(np.int32)
        y = np.stack([data[s+1:s+context+1] for s in starts]).astype(np.int32)
        yield mx.array(x), mx.array(y)


def loss_fn(model, x, y):
    logits, _ = model(x)
    return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")


def evaluate(model, data, batch, context, count, seed):
    rng = np.random.default_rng(seed); losses=[]
    for x, y in batches(data, batch, context, count, rng):
        loss = loss_fn(model, x, y); mx.eval(loss); losses.append(float(loss))
    return float(np.mean(losses))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph", default="synthetic")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("runs/latest"))
    p.add_argument("--neurons", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--recurrent-steps", type=int, default=1)
    p.add_argument("--edge-threshold", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=7)
    args=p.parse_args(); mx.random.seed(args.seed)
    meta=json.loads((args.data_dir/"meta.json").read_text())
    train=np.load(args.data_dir/"train.npy", mmap_mode="r")
    val=np.load(args.data_dir/"val.npy", mmap_mode="r")
    ids, adjacency=build_graph(args.graph,args.neurons,args.edge_threshold,args.seed)
    model=ConnectomeLM(adjacency,meta["vocab_size"],args.embedding_dim,args.recurrent_steps)
    optimizer=optim.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)
    loss_and_grad=nn.value_and_grad(model,loss_fn)
    args.output.mkdir(parents=True,exist_ok=True); np.save(args.output/"neuron_ids.npy",ids)
    np.save(args.output/"adjacency.npy",adjacency)
    rng=np.random.default_rng(args.seed); history=[]; best=math.inf; start=time.perf_counter()
    stream=batches(train,args.batch_size,args.context_length,args.iterations,rng)
    for iteration,(x,y) in enumerate(stream,1):
        loss,grads=loss_and_grad(model,x,y); optimizer.update(model,grads); mx.eval(model.parameters(),optimizer.state)
        if iteration==1 or iteration%args.eval_every==0 or iteration==args.iterations:
            val_loss=evaluate(model,val,args.batch_size,args.context_length,8,args.seed+iteration)
            row={"iteration":iteration,"train_loss":float(loss),"val_loss":val_loss,
                 "val_perplexity":math.exp(val_loss),"seconds":time.perf_counter()-start}
            history.append(row); print(json.dumps(row))
            if val_loss<best:
                best=val_loss; model.save_weights(str(args.output/"best.safetensors"))
                config=vars(args)|{"data_dir":str(args.data_dir),"output":str(args.output),
                                   "vocab_size":meta["vocab_size"],"backend":"mlx","best_val_loss":best}
                (args.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    (args.output/"metrics.json").write_text(json.dumps({"history":history,"best_val_loss":best},indent=2)+"\n")


if __name__=="__main__": main()
