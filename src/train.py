import torch
from torch_geometric.nn.models import LightGCN
from torch_geometric.loader import NeighborLoader
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.model import bpr_loss, BPRChunkDataset, evaluate_faiss, infer_embeddings
from tqdm.auto import tqdm
from torch_sparse import SparseTensor

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

def train():
    # hyperparams
    num_epochs = 50
    batch_size = 256
    lr = 0.01
    fanout = [5, 5, 5]  # neighbor sampling fanout

    # load data and mappings
    graph_data = torch.load('data/graph.pt', weights_only=False, map_location='cpu')
    # edge_index_cpu = graph_data.edge_index.clone()  # clone to avoid modifying original
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
    user_idx = torch.from_numpy(val_df['u_idx'].values).long()
    item_idx = torch.from_numpy(val_df['s_idx'].values).long()
    val_dataset = TensorDataset(user_idx, item_idx)

    # Copy to GPU
    graph_data = graph_data.to(device='cuda')
    print("Graph data moved to GPU.")

    # init model and optimizer
    model_gpu = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).to(device='cuda')
    # model_cpu = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).cpu()  # for CPU eval
    # model_cpu.load_state_dict(model_gpu.state_dict())  # copy weights to CPU model
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

    # setup BPR loader
    bpr_dataset = BPRChunkDataset('data/bpr_triples_chunks', shuffle=True)
    bpr_loader = DataLoader(
        bpr_dataset, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True
    )
    print("BPR dataset and loader ready.")

    # setup val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=4,
    )
    print("Validation DataLoader ready.")

    # prepare for training
    best_hr = 0.0
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs"):
        model_gpu.train()
        epoch_loss = 0.0
        
        # iterate over batches
        triple_iter = iter(bpr_loader)
        for batch_data in train_loader:
            try:
                users, pos, neg = next(triple_iter)
            except StopIteration:
                break
            users = users.to('cuda', non_blocking=True)
            pos = pos.to('cuda', non_blocking=True)
            neg = neg.to('cuda', non_blocking=True)
            # print(f"Processing batch: users={users.shape}, pos={pos.shape}, neg={neg.shape}", flush=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                # forward pass
                embeddings = model_gpu.get_embedding(batch_data.edge_index)
                loss = bpr_loss(users, pos, neg, embeddings)
            # print("Forward pass complete, loss computed.", flush=True)
            # backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # print("Backward pass complete, optimizer step done.", flush=True)

        # FAISS-based evaluation
        print("Syncing to CPU for eval", flush=True)
        torch.cuda.empty_cache()
        emb_dim = model_gpu.embedding_dim
        emb_full_gpu = infer_embeddings(
            model_gpu,
            graph_data.edge_index,   # already on GPU
            num_nodes = num_users + num_items,
            emb_dim    = emb_dim,
            batch_size=4096          # tune up or down as needed
        )
        emb_full = emb_full_gpu.cpu()      
        # Slice embeddings into numpy arrays (float32 for FAISS)
        user_emb = emb_full[:num_users].numpy().astype('float32')
        item_emb = emb_full[num_users:].numpy().astype('float32')
        # model_cpu.load_state_dict(model_gpu.state_dict())  # sync weights to CPU model
        hr10 = evaluate_faiss(
            model_gpu, 
            val_loader, 
            item_emb=item_emb, 
            user_emb=user_emb, 
            top_k=10
        )
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | HR@10: {hr10:.4f}", flush=True)

        # checkpoint
        if hr10 > best_hr:
            best_hr = hr10
            torch.save(model_gpu.state_dict(), 'model.pth')
            print(f"New best HR@10: {best_hr:.4f}, model saved.")

    print(f"Training complete. Best HR@10: {best_hr:.4f}")

if __name__ == "__main__":
    train()