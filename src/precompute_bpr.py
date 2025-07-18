import os
import torch
import random
import pandas as pd
from torch_sparse import SparseTensor
from tqdm import tqdm
import psutil

# Paths
GRAPH_PATH    = 'data/graph.pt'
MAP_PATH      = 'data/mappings.pt'
OUTPUT_DIR    = 'data/bpr_triples_chunks'
CHUNK_SIZE    = 10_000  # users per chunk
NUM_NEG       = 1       # negatives per positive

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Monitor process
process = psutil.Process(os.getpid())

def log_mem(stage):
    rss = process.memory_info().rss / (1024**3)
    print(f"[MEM] {stage}: {rss:.2f} GiB")

# 1) Load graph & mappings on CPU
log_mem('start')
print("Loading graph and mappings...")
graph_data = torch.load(GRAPH_PATH, map_location='cpu')
maps       = torch.load(MAP_PATH, map_location='cpu')
num_users  = len(maps['user2idx'])
num_items  = len(maps['item2idx'])
num_nodes  = num_users + num_items
print(f"Graph: {num_nodes} nodes, {graph_data.num_edges} edges")

# 2) Build CSR representation
print("Building CSR...")
src, dst = graph_data.edge_index
adj_t = SparseTensor(row=src, col=dst, sparse_sizes=(num_nodes, num_nodes))
rowptr, col, _ = adj_t.csr()
log_mem('after CSR')

# 3) Precompute in user chunks
total_users = num_users
for start in tqdm(range(0, total_users, CHUNK_SIZE), desc="Precomputing BPR chunks"):
    end = min(start + CHUNK_SIZE, total_users)
    users_chunk = range(start, end)

    out_u, out_pos, out_neg = [], [], []
    for u in users_chunk:
        s, e = rowptr[u].item(), rowptr[u+1].item()
        neigh = col[s:e]
        pos_tensor = neigh[neigh >= num_users] - num_users
        if pos_tensor.numel() == 0:
            continue
        pos_list = pos_tensor.tolist()
        for pos_local in pos_list:
            for _ in range(NUM_NEG):
                neg = random.randrange(num_items)
                # small rejection set
                while neg in pos_list:
                    neg = random.randrange(num_items)
                out_u.append(u)
                out_pos.append(pos_local + num_users)
                out_neg.append(neg + num_users)

    # Convert to tensors and save chunk
    users_tensor = torch.tensor(out_u, dtype=torch.long)
    pos_tensor   = torch.tensor(out_pos, dtype=torch.long)
    neg_tensor   = torch.tensor(out_neg, dtype=torch.long)
    chunk_file = os.path.join(OUTPUT_DIR, f'bpr_{start}_{end}.pt')
    torch.save({'users': users_tensor,
                'pos':   pos_tensor,
                'neg':   neg_tensor}, chunk_file)
    log_mem(f'after saving chunk {start}-{end}')

print("Precompute complete.")
