import torch
from torch_geometric.nn.models import LightGCN
from torch_geometric.loader import NeighborLoader
from torch.amp import autocast, GradScaler
import pandas as pd
from src.model import bpr_loss, sample_bpr_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_hr10(model, graph_data, val_interactions, mappings):
    """
    Compute HR@10 over val_interactions (DataFrame with u_idx, s_idx).
    For each user, score all items, pick top-10, check hit.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.edge_index.to(device))
        user_emb = embeddings[:mappings['num_users']]
        item_emb = embeddings[mappings['num_users']:]
        scores = user_emb @ item_emb.t()

    hits = 0
    total = 0
    for u, i in zip(val_interactions['u_idx'], val_interactions['s_idx']):
        top10 = torch.topk(scores[u], k=10).indices.tolist()
        if (i - mappings['num_users']) in top10:
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
    graph_data = torch.load('data/graph.pt', weights_only=False)
    graph_data = graph_data.to(device)
    maps = torch.load('data/mappings.pt', weights_only=False)
    num_users = len(maps['user2idx'])
    num_items = len(maps['item2idx'])
    val_df = pd.read_csv('data/val_triplets.txt', sep='\t', header=None, names=['user', 'song', 'playcount', 'u_idx', 's_idx'])

    # build user2items mapping
    src, dst = graph_data.edge_index
    user2items = {u: [] for u in range(num_users)}
    for u, i in zip(src.tolist(), dst.tolist()):
        if u < num_users:
            user2items[u].append(i)

    # init model and optimizer
    model = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    # set up neighbor sampler
    train_loader = NeighborLoader(
        data=graph_data,
        num_neighbors=fanout,
        batch_size=batch_size,
        input_nodes=torch.arange(num_users, device=device),  # only sample users
        shuffle=True,
    )

    best_hr = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        # iterate over mini-batches
        for batch_data in train_loader:
            batch_users = batch_data.n_id[: batch_data.batch_size]  # user indices in this batch
            edge_index_sub = batch_data.edge_index

            # randomly sample positive & negative items for these users
            _, pos, neg = sample_bpr_batch(
                batch_users, 
                user2items, 
                num_users, 
                num_items
            )

            with autocast():
                # forward pass
                embeddings = model(edge_index_sub.to(device))
                loss = bpr_loss(batch_users, pos.to(device), neg.to(device), embeddings)
            
            # backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # validation
        hr10 = evaluate_hr10(model, graph_data, val_df, {
            'num_users': num_users,
            'num_items': num_items
        })
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | HR@10: {hr10:.4f}")

        # checkpoint
        if hr10 > best_hr:
            best_hr = hr10
            torch.save(model.state_dict(), 'model.pth')
            print(f"New best HR@10: {best_hr:.4f}, model saved.")

    print(f"Training complete. Best HR@10: {best_hr:.4f}")

if __name__ == "__main__":
    train()