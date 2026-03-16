import sys
import os
import torch
import pickle
import numpy as np
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

from model import GPTConfig, GPT


def get_args():
    parser = argparse.ArgumentParser(description="LSE-MTP Representation Alignment (Structure Gain) Test")

    parser.add_argument('--dataset', type=str, default='usg')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'lse'])
    parser.add_argument('--n_tokens', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--ckpt_iter', type=str, default='final', help="final or specific iteration")

    parser.add_argument('--checkpoint', type=str, default='', help="Manual path to .pt file")
    parser.add_argument('--k', type=int, default=3, help="Metric Horizon k")
    parser.add_argument('--max_samples', type=int, default=1000, help="Number of paths to sample")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducible sampling")

    return parser.parse_args()


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return None, "N/A"

    checkpoint = torch.load(ckpt_path, map_location=device)
    conf = GPTConfig(**checkpoint['model_args'])
    model = GPT(conf)

    sd = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in sd.keys()):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    model.load_state_dict(sd)
    model.to(device).eval()

    if conf.n_tokens == 1:
        method = "NTP"
    elif conf.latent_lambda > 0:
        method = "LSE-MTP"
    else:
        method = "MTP"

    return model, method


def extract_states(model, stoi, test_file, device, max_samples, k):
    states, labels_next, labels_future = [], [], []
    temp_activation = {}

    def hook(model, input, output):
        temp_activation['h'] = output.detach()

    handle = model.transformer.ln_f.register_forward_hook(hook)

    with open(test_file, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]

    sample_lines = random.sample(lines, min(len(lines), max_samples))

    for line in tqdm(sample_lines, desc="Extracting States", leave=False):
        parts = line.split()
        if len(parts) < 5: continue

        tokens = [stoi[parts[0]], stoi[parts[1]]]
        for inc_str in parts[2:]:
            if inc_str in stoi:
                tokens.append(stoi[inc_str])

        for i in range(2, len(tokens) - k):
            input_ids = torch.tensor(tokens[:i + 1], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad(): model(input_ids)

            states.append(temp_activation['h'][0, -1, :].cpu().numpy())
            labels_next.append(tokens[i + 1])
            labels_future.append(tokens[i + k])

    handle.remove()
    return np.array(states), np.array(labels_next), np.array(labels_future)


def compute_gain(states, next_labels, future_labels):
    if len(states) == 0: return 0, 0, 0
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    states_norm = states / (norms + 1e-8)

    fut_to_idx = defaultdict(list)
    for idx, f_lab in enumerate(future_labels): fut_to_idx[f_lab].append(idx)

    aligned_sims = []
    for indices in fut_to_idx.values():
        if len(indices) < 2: continue
        sub = random.sample(indices, min(len(indices), 100))
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                if next_labels[sub[i]] != next_labels[sub[j]]:
                    aligned_sims.append(np.dot(states_norm[sub[i]], states_norm[sub[j]]))

    random_sims = []
    for _ in range(min(len(aligned_sims), 5000) if aligned_sims else 5000):
        idx_a, idx_b = random.randint(0, len(states) - 1), random.randint(0, len(states) - 1)
        if future_labels[idx_a] != future_labels[idx_b]:
            random_sims.append(np.dot(states_norm[idx_a], states_norm[idx_b]))

    s_fut = np.mean(aligned_sims) if aligned_sims else 0
    s_rnd = np.mean(random_sims) if random_sims else 0
    return s_fut, s_rnd, s_fut - s_rnd


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        method_prefix = "lse_" if args.method == 'lse' else ""
        folder_name = f"{method_prefix}{args.dataset}_{args.n_layer}_{args.n_head}_{args.n_embd}_{args.num_nodes}_{args.n_tokens}t"
        ckpt_filename = 'final_model.pt' if args.ckpt_iter == 'final' else f'ckpt_{args.ckpt_iter}.pt'
        ckpt_path = os.path.join(project_root, "out", folder_name, ckpt_filename)

    data_dir = os.path.join(project_root, "data", f"{args.dataset}_{args.num_nodes}")
    test_file = os.path.join(data_dir, 'val_incremental.txt')
    meta_path = os.path.join(data_dir, 'meta_incremental.pkl')

    print(f"Loading checkpoint: {ckpt_path}")
    model, method_name = load_model(ckpt_path, args.device)
    if model is None:
        print(f"Error: Model file not found at {ckpt_path}")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    print(f"Evaluating Structure Gain (k={args.k}) on {args.max_samples} samples...")
    states, n_lbl, f_lbl = extract_states(model, meta['stoi'], test_file, args.device, args.max_samples, args.k)

    sim_fut, sim_rnd, gain = compute_gain(states, n_lbl, f_lbl)

    print("\n" + "=" * 60)
    print(f"ALIGNMENT REPORT: {os.path.basename(ckpt_path)}")
    print("-" * 60)
    print(f"Method:       {method_name}")
    print(f"Sim(Future):  {sim_fut:.4f}  (Similarity: same future, different actions)")
    print(f"Sim(Random):  {sim_rnd:.4f}  (Random state similarity)")
    print(f"STRUCTURE GAIN: {gain:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()