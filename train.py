import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from model import GPT, GPTConfig

try:
    from logger import get_logger
except ImportError:
    import logging
    def get_logger(path):
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(path), logging.StreamHandler()])
        return logging.getLogger()

def train_model():
    parser = argparse.ArgumentParser(description="LSE-MTP Training Script")
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='usg', help="er or usg")
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'lse'])
    parser.add_argument('--n_tokens', type=int, default=1, help="Horizon K")
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--lambda_l', type=float, default=0.1, help="Latent loss weight")
    parser.add_argument('--lambda_s', type=float, default=0.1, help="Semantic loss weight")
    parser.add_argument('--max_iters', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    # Directory Setup
    method_prefix = "lse_" if args.method == 'lse' else ""
    out_dir = f'out/{method_prefix}{args.dataset}_{args.n_layer}_{args.n_head}_{args.n_embd}_{args.num_nodes}_{args.n_tokens}t'
    data_dir = os.path.join('data', f'{args.dataset}_{args.num_nodes}')

    # DDP Initialization
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = int(os.environ['RANK']) == 0
        seed_offset = int(os.environ['RANK'])
    else:
        master_process = True
        seed_offset = 0
        device = 'cuda'

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Data directory: {data_dir}")

    torch.manual_seed(1337 + seed_offset)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    # Metadata Loading
    meta_path = os.path.join(data_dir, 'meta_incremental.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Check --dataset and --num_nodes.")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    data_size = block_size + 1

    # Data Loading (Memory Map)
    train_data = np.memmap(os.path.join(data_dir, 'train_incremental.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val_incremental.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        b_size = 1024 if split == 'train' else 64

        # Aligned block sampling
        num_blocks = (len(data) - (data_size + args.n_tokens)) // data_size
        ix = torch.randint(num_blocks, (b_size,)) * data_size

        x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix])

        # Construct multi-horizon target matrix (Batch, K, Seq)
        y_list = []
        for k in range(1, args.n_tokens + 1):
            y_k = torch.stack([torch.from_numpy((data[i + k: i + k + block_size]).astype(np.int64)) for i in ix])
            # Mask tokens that are out of sequence bounds
            if k > 1:
                for m in range(1, k):
                    y_k[:, -m] = 0
            y_list.append(y_k)

        Y = torch.stack(y_list, dim=1)

        if device_type == 'cuda':
            x, Y = x.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
        return x, Y

    # Model Initialization
    lat_lambda = args.lambda_l if args.method == 'lse' else 0.0
    sem_lambda = args.lambda_s if args.method == 'lse' else 0.0

    gptconf = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=block_size, vocab_size=vocab_size,
        n_tokens=args.n_tokens, latent_lambda=lat_lambda, semantic_lambda=sem_lambda,
        bias=False, dropout=0.0
    )
    model = GPT(gptconf).to(device)

    if master_process:
        print(f"Compiling model for {args.method} (K={args.n_tokens})...")
        model = torch.compile(model)
        logger = get_logger(os.path.join(out_dir, "train.log"))

    optimizer = model.configure_optimizers(0.1, args.lr, (0.9, 0.95), device_type)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Learning Rate Schedule
    def get_lr(it):
        warmup = args.max_iters // 20
        if it < warmup: return args.lr * it / warmup
        if it > args.max_iters: return args.lr / 10
        ratio = (it - warmup) / (args.max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return (args.lr / 10) + coeff * (args.lr - args.lr / 10)

    # Training Loop
    iter_num = 0
    X, Y = get_batch('train')
    raw_model = model.module if ddp else model

    if master_process: pbar = tqdm(total=args.max_iters, desc="Training")

    while iter_num <= args.max_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        if iter_num > 0 and iter_num % 5000 == 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': vars(gptconf),
                'iter_num': iter_num,
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

        model.train()
        with ctx:
            _, losses = model(X, Y)
            loss = sum(losses)

        X, Y = get_batch('train')

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if master_process:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")
            pbar.update(1)

        iter_num += 1

    if master_process:
        torch.save({'model': raw_model.state_dict(), 'model_args': vars(gptconf)},
                   os.path.join(out_dir, 'final_model.pt'))
        print("Training complete.")

    if ddp: destroy_process_group()

if __name__ == "__main__":
    train_model()