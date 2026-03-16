import os
import random
import argparse
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial import Delaunay
from itertools import islice

def gen_er_graph(n, p):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < p:
                G.add_edge(i, j, weight=1.0)
    return G

def gen_usg_graph(n, p_density):
    pos = {i: (random.random(), random.random()) for i in range(n)}
    points = np.array([pos[i] for i in range(n)])
    tri = Delaunay(points)
    base_graph = nx.Graph()

    for i in range(n):
        base_graph.add_node(i, pos=pos[i])

    for simplex in tri.simplices:
        for k in range(3):
            u, v = simplex[k], simplex[(k + 1) % 3]
            dist = np.linalg.norm(points[u] - points[v])
            base_graph.add_edge(u, v, weight=dist)

    mst = nx.minimum_spanning_tree(base_graph)
    city_graph = nx.Graph()
    city_graph.add_nodes_from(base_graph.nodes(data=True))
    city_graph.add_edges_from(mst.edges())

    non_mst_edges = [e for e in base_graph.edges() if not mst.has_edge(*e)]
    random.shuffle(non_mst_edges)
    for i in range(int(len(non_mst_edges) * p_density)):
        city_graph.add_edge(*non_mst_edges[i])

    nodes_sorted = sorted(list(city_graph.nodes(data='pos')), key=lambda x: (x[1][0], x[1][1]))
    mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(nodes_sorted)}
    city_graph = nx.relabel_nodes(city_graph, mapping)
    return city_graph.to_directed()

def get_augmented_paths(G, pairs, args):
    all_paths = []
    for u, v in tqdm(pairs, desc="Generating paths"):
        if not nx.has_path(G, u, v): continue
        try:
            paths = list(islice(nx.shortest_simple_paths(G, u, v, weight='weight'), args.k_paths))
            for p in paths: all_paths.append({'src': u, 'tgt': v, 'path': p})
        except:
            pass

        if random.random() < args.detour_prob:
            try:
                shortest = nx.shortest_path(G, u, v, weight='weight')
                if len(shortest) > 3:
                    G_tmp = G.copy()
                    G_tmp.remove_node(random.choice(shortest[1:-1]))
                    if nx.has_path(G_tmp, u, v):
                        all_paths.append({'src': u, 'tgt': v, 'path': nx.shortest_path(G_tmp, u, v)})
            except:
                pass

        if random.random() < args.recovery_prob:
            try:
                neighbors = list(G.successors(u))
                if len(neighbors) > 1:
                    wrong_step = random.choice(neighbors)
                    if nx.has_path(G, wrong_step, v):
                        all_paths.append({'src': u, 'tgt': v, 'path': [u] + nx.shortest_path(G, wrong_step, v)})
            except:
                pass
    return all_paths

def save_as_incremental_bin(path_data, num_nodes, output_dir, split):
    stoi = {'[PAD]': 0, '\n': 1}
    for i in range(num_nodes): stoi[str(i)] = len(stoi)
    for i in range(-(num_nodes - 1), num_nodes):
        if i == 0: continue
        stoi[f"inc_{i}"] = len(stoi)

    itos = {v: k for k, v in stoi.items()}
    processed_lines = []
    max_len = 0
    for item in path_data:
        src, tgt, p = item['src'], item['tgt'], item['path']
        line = [stoi[str(src)], stoi[str(tgt)]]
        for i in range(len(p) - 1):
            line.append(stoi[f"inc_{p[i + 1] - p[i]}"])
        line.append(1)
        processed_lines.append(line)
        max_len = max(max_len, len(line))

    block_size = ((max_len - 1) // 32 + 1) * 32
    final_ids = []
    for line in processed_lines:
        padding = [0] * (block_size + 1 - len(line))
        final_ids.extend(line + padding)

    arr = np.array(final_ids, dtype=np.uint16)
    arr.tofile(os.path.join(output_dir, f"{split}_incremental.bin"))
    return stoi, itos, block_size

def save_as_txt(path_data, output_dir, split):
    with open(os.path.join(output_dir, f"{split}_incremental.txt"), 'w') as f:
        for item in path_data:
            line = f"{item['src']} {item['tgt']} "
            line += " ".join([f"inc_{item['path'][i + 1] - item['path'][i]}" for i in range(len(item['path']) - 1)])
            f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='usg', choices=['er', 'usg'], help="Graph topology type")
    parser.add_argument('--nodes', type=int, default=100, help="Number of nodes")
    parser.add_argument('--edge_p', type=float, default=0.3, help="Edge probability (ER) or density (USG)")
    parser.add_argument('--train_ratio', type=float, default=0.5, help="Ratio of training pairs")
    parser.add_argument('--k_paths', type=int, default=3, help="Number of shortest paths per pair")
    parser.add_argument('--detour_prob', type=float, default=0.3, help="Probability of generating detour paths")
    parser.add_argument('--recovery_prob', type=float, default=0.3, help="Probability of generating recovery paths")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    data_dir = f"data/{args.type}_{args.nodes}"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Generating {args.type.upper()} graph...")
    G = gen_er_graph(args.nodes, args.edge_p) if args.type == 'er' else gen_usg_graph(args.nodes, args.edge_p)

    all_pairs = [(u, v) for u in G.nodes() for v in G.nodes() if u != v and nx.has_path(G, u, v)]
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * args.train_ratio)

    print(f"Sampling augmented paths...")
    train_paths = get_augmented_paths(G, all_pairs[:split_idx], args)
    val_paths = [{'src': u, 'tgt': v, 'path': nx.shortest_path(G, u, v)} for u, v in all_pairs[split_idx:]]

    print(f"Serializing files...")
    stoi, itos, block_size = save_as_incremental_bin(train_paths, args.nodes, data_dir, 'train')
    _ = save_as_incremental_bin(val_paths, args.nodes, data_dir, 'val')

    save_as_txt(train_paths, data_dir, 'train')
    save_as_txt(val_paths, data_dir, 'val')

    meta = {'stoi': stoi, 'itos': itos, 'block_size': block_size, 'vocab_size': len(stoi)}
    with open(os.path.join(data_dir, "meta_incremental.pkl"), "wb") as f:
        pickle.dump(meta, f)

    G_clean = G.copy()
    for n in G_clean.nodes():
        if 'pos' in G_clean.nodes[n]:
            del G_clean.nodes[n]['pos']
    nx.write_graphml(G_clean, os.path.join(data_dir, "graph.graphml"))

    print("-" * 30)
    print(f"Success! Files saved in {data_dir}")