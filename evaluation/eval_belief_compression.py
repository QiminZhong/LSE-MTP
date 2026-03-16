import sys
import os
import torch
import torch.nn.functional as F
import pickle
import networkx as nx
import numpy as np
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

from model import GPTConfig, GPT

def get_args():
    parser = argparse.ArgumentParser(description="LSE-MTP Belief Compression & Goal Distinction Test")

    parser.add_argument('--dataset', type=str, default='usg')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'lse'])
    parser.add_argument('--n_tokens', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--ckpt_iter', type=str, default='final', help="final or specific iteration")

    parser.add_argument('--checkpoint', type=str, default='', help="Manual path to .pt file")
    parser.add_argument('--dataset_dir', type=str, default='', help="Manual path to dataset directory")

    parser.add_argument('--num_trials', type=int, default=1000, help="Number of trials per group")
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

def sample_path_reverse(G, reverse_adj, end_node, length):
    path = [end_node]
    curr = end_node
    for _ in range(length):
        predecessors = reverse_adj[curr]
        if not predecessors: return None
        prev = random.choice(predecessors)
        path.append(prev)
        curr = prev
    path.reverse()
    return path

def find_test_groups(G, num_trials):
    reverse_adj = defaultdict(list)
    for u, v in G.edges():
        reverse_adj[v].append(u)

    nodes = list(G.nodes())
    groups = {
        'A_SameG_SameP': [],
        'B_DiffG_DiffP': [],
        'C_SameG_DiffP': [],
        'D_DiffG_SameP': []
    }

    print(f"Generating test groups ({num_trials} trials each)...")
    pbar = tqdm(total=num_trials * 4)

    while len(groups['A_SameG_SameP']) < num_trials or len(groups['D_DiffG_SameP']) < num_trials:
        p_node = random.choice(nodes)
        p1 = sample_path_reverse(G, reverse_adj, p_node, random.randint(3, 6))
        p2 = sample_path_reverse(G, reverse_adj, p_node, random.randint(3, 6))
        if not p1 or not p2 or p1[0] == p2[0]: continue

        targets = [n for n in nodes if nx.has_path(G, p_node, n) and n != p_node]
        if len(targets) >= 2:
            t1, t2 = random.sample(targets, 2)
            if len(groups['A_SameG_SameP']) < num_trials:
                groups['A_SameG_SameP'].append({'p1': p1, 'p2': p2, 't1': t1, 't2': t1})
                pbar.update(1)
            if len(groups['D_DiffG_SameP']) < num_trials:
                groups['D_DiffG_SameP'].append({'p1': p1, 'p2': p2, 't1': t1, 't2': t2})
                pbar.update(1)

    while len(groups['B_DiffG_DiffP']) < num_trials or len(groups['C_SameG_DiffP']) < num_trials:
        n1, n2 = random.sample(nodes, 2)
        p1 = sample_path_reverse(G, reverse_adj, n1, random.randint(2, 4))
        p2 = sample_path_reverse(G, reverse_adj, n2, random.randint(2, 4))
        if not p1 or not p2: continue

        common_ts = [n for n in nodes if nx.has_path(G, n1, n) and nx.has_path(G, n2, n)]
        if common_ts and len(groups['C_SameG_DiffP']) < num_trials:
            t = random.choice(common_ts)
            groups['C_SameG_DiffP'].append({'p1': p1, 'p2': p2, 't1': t, 't2': t})
            pbar.update(1)

        t1_list = [n for n in nodes if nx.has_path(G, n1, n)]
        t2_list = [n for n in nodes if nx.has_path(G, n2, n)]
        if t1_list and t2_list:
            t1, t2 = random.choice(t1_list), random.choice(t2_list)
            if t1 != t2 and len(groups['B_DiffG_DiffP']) < num_trials:
                groups['B_DiffG_DiffP'].append({'p1': p1, 'p2': p2, 't1': t1, 't2': t2})
                pbar.update(1)

    pbar.close()
    return groups

def get_latent_rep(model, stoi, source, target, path, device):
    tokens = [stoi[str(source)], stoi[str(target)]]
    curr = path[0]
    for nxt in path[1:]:
        tokens.append(stoi[f"inc_{nxt - curr}"])
        curr = nxt

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    activation = {}
    def hook(model, input, output): activation['h'] = output.detach()
    handle = model.transformer.ln_f.register_forward_hook(hook)

    with torch.no_grad(): model(x)

    handle.remove()
    return activation['h'][0, -2, :]

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

    data_dir = args.dataset_dir if args.dataset_dir else os.path.join(project_root, "data", f"{args.dataset}_{args.num_nodes}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, method_name = load_model(ckpt_path, args.device)
    if model is None:
        print("Error: Model not found.")
        return

    with open(os.path.join(data_dir, 'meta_incremental.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']

    G = nx.read_graphml(os.path.join(data_dir, 'graph.graphml'))
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})

    test_groups = find_test_groups(G, args.num_trials)

    print(f"Evaluating belief states for {method_name}...")
    group_sims = {}
    for g_name, cases in test_groups.items():
        sims = []
        for c in tqdm(cases, desc=f"  {g_name}", leave=False):
            h1 = get_latent_rep(model, stoi, c['p1'][0], c['t1'], c['p1'], args.device)
            h2 = get_latent_rep(model, stoi, c['p2'][0], c['t2'], c['p2'], args.device)
            sims.append(F.cosine_similarity(h1, h2, dim=0).item())
        group_sims[g_name] = np.mean(sims)

    print("\n" + "=" * 95)
    print(f"BELIEF COMPRESSION REPORT: {os.path.basename(ckpt_path)}")
    print("-" * 95)
    print(f"{'SameG_SameP':<15} | {'SameG_DiffP':<15} | {'DiffG_SameP':<15} | {'Base(DiffG_DiffP)':<15}")
    print("-" * 95)
    s = group_sims
    print(f"{s['A_SameG_SameP']:>15.4f} | {s['C_SameG_DiffP']:>15.4f} | {s['D_DiffG_SameP']:>15.4f} | {s['B_DiffG_DiffP']:>15.4f}")
    print("=" * 95)

if __name__ == "__main__":
    main()