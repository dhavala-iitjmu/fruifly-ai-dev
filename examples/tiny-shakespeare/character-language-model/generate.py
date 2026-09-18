"""Generate text from a trained MLX connectome language model."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import mlx.core as mx
import numpy as np
from model import ConnectomeLM


def load(run: Path):
    cfg=json.loads((run/"config.json").read_text())
    meta=json.loads((Path(cfg["data_dir"])/"meta.json").read_text())
    adj=np.load(run/"adjacency.npy")
    model=ConnectomeLM(adj,cfg["vocab_size"],cfg["embedding_dim"],cfg["recurrent_steps"])
    model.load_weights(str(run/"best.safetensors")); model.eval()
    return model,meta,cfg


def generate(model,meta,prompt,length,temperature,top_k,seed):
    mx.random.seed(seed); stoi={c:i for i,c in enumerate(meta["characters"])}; itos=meta["characters"]
    unknown=[c for c in prompt if c not in stoi]
    if unknown: raise ValueError(f"Characters outside vocabulary: {unknown}")
    tokens=[stoi[c] for c in prompt]; state=None
    logits,state=model(mx.array([tokens],dtype=mx.int32),state)
    # Compile the fixed-shape, one-character recurrent step. Without this,
    # hundreds of tiny eager Metal dispatches dominate generation time.
    step = mx.compile(lambda token, hidden: model(token, hidden))
    for _ in range(length):
        scores=logits[0,-1]/max(temperature,1e-6)
        if top_k>0:
            k=min(top_k,len(itos)); cutoff=mx.sort(scores)[-k]; scores=mx.where(scores<cutoff,-mx.inf,scores)
        nxt=int(mx.random.categorical(scores))
        tokens.append(nxt); logits,state=step(mx.array([[nxt]],dtype=mx.int32),state)
    return ''.join(itos[i] for i in tokens)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,default=Path("runs/latest"))
    p.add_argument("--prompt",default="ROMEO:\n"); p.add_argument("--characters",type=int,default=500)
    p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=20)
    p.add_argument("--seed",type=int,default=7); a=p.parse_args()
    model,meta,_=load(a.run); print(generate(model,meta,a.prompt,a.characters,a.temperature,a.top_k,a.seed))


if __name__=="__main__": main()
