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

from model import GPTConfig, GPT

def get_args():
    parser = argparse.ArgumentParser(description="LSE-MTP Structural Hallucination (ISP) Test")

    parser.add_argument('--dataset', type=str, default='usg')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--method', type=str, default='lse', choices=['standard', 'lse'])
    parser.add_argument('--n_tokens', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--ckpt_iter', type=str, default='final', help="final or specific iteration")

    parser.add_argument('--checkpoint', type=str, default='', help="Manual path to .pt file (optional)")
    parser.add_argument('--dataset_dir', type=str, default='', help="Manual path to dataset directory (optional)")

    parser.add_argument('--max_samples', type=int, default=5000, help="Number of (A, B, G) structures to mine")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help="Seed to ensure same test structures across runs")

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
        method_name = "NTP"
    elif conf.latent_lambda > 0:
        method_name = "LSE-MTP"
    else:
        method_name = "MTP"

    return model, method_name

def find_hallucination_candidates(G, max_samples=1000):
    nodes = list(G.nodes())
    candidates = []
    print(f"Mining structural candidates (A, B, G) from graph... Target: {max_samples}")
    pbar = tqdm(total=max_samples)

    attempts = 0
    while len(candidates) < max_samples and attempts < max_samples * 50:
        attempts += 1
        g = random.choice(nodes)

        preds_dict = nx.single_source_shortest_path_length(G.reverse(), g, cutoff=4)
        potential_nodes = [n for n, dist in preds_dict.items() if 2 <= dist <= 4]

        if len(potential_nodes) < 2: continue

        a, b = random.sample(potential_nodes, 2)

        if not G.has_edge(a, b):
            a_successors = list(G.successors(a))
            if not a_successors: continue

            candidates.append({
                'a': a,
                'b': b,
                'g': g,
                'illegal_inc': b - a,
                'legal_incs': [s - a for s in a_successors]
            })
            pbar.update(1)

    pbar.close()
    return candidates

@torch.no_grad()
def evaluate_model_isp(model, stoi, candidates, device):
    isp_list = []
    legal_sum_list = []

    for c in tqdm(candidates, desc="Evaluating", leave=False):
        x = torch.tensor([stoi[str(c['a'])], stoi[str(c['g'])]], dtype=torch.long, device=device).unsqueeze(0)

        logits_tuple, _ = model(x)
        logits = logits_tuple[0]
        probs = F.softmax(logits[0, -1, :], dim=-1)

        ill_token = f"inc_{c['illegal_inc']}"
        p_illegal = probs[stoi[ill_token]].item() if ill_token in stoi else 0.0

        p_legal = 0.0
        for l_inc in c['legal_incs']:
            l_token = f"inc_{l_inc}"
            if l_token in stoi:
                p_legal += probs[stoi[l_token]].item()

        isp_list.append(p_illegal)
        legal_sum_list.append(p_legal)

    return np.mean(isp_list), np.mean(legal_sum_list)

def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        method_prefix = "lse_" if args.method == 'lse' else ""
        folder_name = f"{method_prefix}{args.dataset}_{args.n_layer}_{args.n_head}_{args.n_embd}_{args.num_nodes}_{args.n_tokens}t"
        out_dir = os.path.join(project_root, "out", folder_name)
        ckpt_filename = 'final_model.pt' if args.ckpt_iter == 'final' else f'ckpt_{args.ckpt_iter}.pt'
        ckpt_path = os.path.join(out_dir, ckpt_filename)

    data_dir = args.dataset_dir if args.dataset_dir else os.path.join(project_root, "data", f"{args.dataset}_{args.num_nodes}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, method_name = load_model(ckpt_path, args.device)
    if model is None:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    meta_path = os.path.join(data_dir, 'meta_incremental.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']

    graph_path = os.path.join(data_dir, 'graph.graphml')
    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})

    candidates = find_hallucination_candidates(G, args.max_samples)

    print(f"Starting ISP Evaluation for {method_name}...")
    avg_isp, avg_legal = evaluate_model_isp(model, stoi, candidates, args.device)

    print("\n" + "=" * 80)
    print(f"STRUCTURAL HALLUCINATION (ISP) REPORT: {os.path.basename(ckpt_path)}")
    print("-" * 80)
    print(f"Method:        {method_name}")
    print(f"Test Samples:  {len(candidates)}")
    print("-" * 80)
    print(f"ISP (Illegal Shortcut Prob) ↓:  {avg_isp:.6f}")
    print(f"Legal Step Probability ↑:      {avg_legal:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()