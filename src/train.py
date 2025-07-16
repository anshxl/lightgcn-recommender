import torch
from torch_geometric.nn.models import LightGCN
from torch_geometric.loader import NeighborLoader
from torch.amp import autocast, GradScaler
import pandas as pd
from src.model import bpr_loss, sample_bpr_batch
from tqdm.auto import tqdm
import random
from torch_sparse import SparseTensor

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

def evaluate_hr10(embeddings, val_df, num_users, num_items, rowptr, col, num_neg=1000):
    """
    embeddings:    [num_users + num_items, D] tensor on CPU
    val_df:        DataFrame with columns ['u_idx','s_idx'] (global item idx)
    num_users:     number of users
    num_items:     number of items
    rowptr, col:   CSR representation of the full training graph (global node ids)
    num_neg:       number of negatives to sample per positive
    """
    user_emb = embeddings[:num_users]       # [U, D]
    item_emb = embeddings[num_users:]       # [I, D]
    hits, total = 0, 0

    # all_items = set(range(num_items))

    for u, global_i in zip(val_df['u_idx'], val_df['s_idx']):
        #shift to 0...I-1
        pos_i = global_i - num_users  # global index to local item index

        # pull this user's training neighbors from CSR
        start, end = rowptr[u].item(), rowptr[u + 1].item()
        neigh_global = col[start:end]     # a torch.LongTensor of global IDs

        # extract only the item-neighbors, and make a set of local indices
        #    (i.e. for each n >= num_users, local = n - num_users)
        pos_set = set(
            (neigh_global[neigh_global >= num_users] - num_users).tolist()
        )

        # sample negatives by rejection until we have num_neg
        negs = []
        while len(negs) < num_neg:
            cand = random.randrange(num_items)
            if cand not in pos_set and cand != pos_i:
                negs.append(cand)

        # build candidate list (positive first, then negatives)
        candidates = [pos_i] + negs

        # score the user against these candidates
        u_vec = user_emb[u].unsqueeze(0)   # [1, D]
        c_vecs = item_emb[torch.tensor(candidates, dtype=torch.long)]
        scores = (u_vec @ c_vecs.t()).squeeze(0)   # [N+1]

        # check if the positive (index 0) is in top-10
        top10 = torch.topk(scores, k=10).indices.tolist()
        if 0 in top10:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0

def train():
    # hyperparams
    num_epochs = 50
    batch_size = 1024
    lr = 0.01
    fanout = [10, 10, 10]  # neighbor sampling fanout

    # load data and mappings
    graph_data = torch.load('data/graph.pt', weights_only=False, map_location='cpu')
    edge_index_cpu = graph_data.edge_index.clone()  # clone to avoid modifying original
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    maps = torch.load('data/mappings.pt', weights_only=False)
    num_users = len(maps['user2idx'])
    num_items = len(maps['item2idx'])
    print(f"Mappings: {num_users} users, {num_items} items")

    val_df = pd.read_csv(
        'data/val_triplets.txt', 
        sep='\t', 
        header=None, 
        names=['user', 'song', 'playcount', 'u_idx', 's_idx']
        ).sample(n=10000, random_state=42).reset_index(drop=True)
    print(f"Validation: {len(val_df)} samples")

    # build CSR on CPU
    num_nodes = num_users + num_items
    src, dst = graph_data.edge_index
    adj_t = SparseTensor(row=src, col=dst, sparse_sizes=(num_nodes, num_nodes))
    rowptr, col_tensor, _ = adj_t.csr()
    print("CSR built: rowptr", rowptr.shape, "col", col_tensor.shape)

    # Copy to GPU
    graph_data = graph_data.to(device='cuda')
    print("Graph data moved to GPU.")

    # init model and optimizer
    model_gpu = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).to(device='cuda')
    model_cpu = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).cpu()  # for CPU eval
    model_cpu.load_state_dict(model_gpu.state_dict())  # copy weights to CPU model
    optimizer = torch.optim.Adam(model_gpu.parameters(), lr=lr)
    scaler = GradScaler()
    print("Model initialized.")

    # set up neighbor sampler
    user_idx = torch.arange(num_users, device='cuda')  # only sample users
    train_loader = NeighborLoader(
        data=graph_data,
        num_neighbors=fanout,
        batch_size=batch_size,
        input_nodes=user_idx,  # only sample users
        shuffle=True,
    )
    print("NeighborLoader ready")

    best_hr = 0.0
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs"):
        model_gpu.train()
        epoch_loss = 0.0
        
        # iterate over batches
        for batch_data in train_loader:
            # move batch data to CPU for sampling
            batch_users_cpu = batch_data.input_id.cpu()
            users_cpu, pos_cpu, neg_cpu = sample_bpr_batch(
                batch_users_cpu, rowptr, col_tensor,
                num_users, num_items, num_neg=1
            )
            if users_cpu.numel() == 0:
                continue
            print(f"Batch size: {batch_users_cpu.shape[0]}, sampled users: {users_cpu.shape[0]}")
            # move tiny samplers to CUDA
            users = users_cpu.to('cuda')
            pos   = pos_cpu.to('cuda')
            neg   = neg_cpu.to('cuda')
            print("Batch data moved to GPU.")

            with autocast(device_type='cuda', dtype=torch.float16):
                # forward pass
                embeddings = model_gpu.get_embedding(batch_data.edge_index)
                loss = bpr_loss(users, pos.to(device='cuda'), neg.to(device='cuda'), embeddings)
            print("Forward pass complete, loss computed.")
            # backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print("Backward pass complete, optimizer step done.")

        print("Syncing to CPU for eval")
        torch.cuda.empty_cache()
        model_cpu.load_state_dict(model_gpu.state_dict())  # sync weights to CPU model
        model_cpu.eval()
        with torch.no_grad():
            emb_full = model_cpu.get_embedding(edge_index_cpu)

        # validation
        hr10 = evaluate_hr10(
            embeddings=emb_full, 
            val_df=val_df, 
            num_users=num_users, 
            num_items=num_items, 
            rowptr=rowptr,
            col=col_tensor, 
            num_neg=1000)
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | HR@10: {hr10:.4f}")

        # checkpoint
        if hr10 > best_hr:
            best_hr = hr10
            torch.save(model_gpu.state_dict(), 'model.pth')
            print(f"New best HR@10: {best_hr:.4f}, model saved.")

    print(f"Training complete. Best HR@10: {best_hr:.4f}")

if __name__ == "__main__":
    train()