import torch
from torch_geometric.nn.models import LightGCN
import pandas as pd
from src.model import bpr_loss, sample_bpr_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_hr10(model, graph_data, val_interactions, mappings):
    """
    Compute HR@10 over val_interactions (DataFrame with u_idx, i_idx).
    For each user, score all items, pick top-10, check hit.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.edge_index)
        user_emb = embeddings[:mappings['num_users']]
        item_emb = embeddings[mappings['num_users']:]
        scores = user_emb @ item_emb.t()

    hits = 0
    total = 0
    for u, i in zip(val_interactions['u_idx'], val_interactions['i_idx']):
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

    # load data and mappings
    graph_data = torch.load('data/graph.pt', weights_only=False)
    graph_data = graph_data.to(device)
    maps = torch.load('data/mappings.pt', weights_only=False)
    num_users = len(maps['user2idx'])
    num_items = len(maps['item2idx'])
    val_df = pd.read_csv('data/val_triplets.txt', sep='\t', header=None, names=['user', 'song', 'playcount', 'u_idx', 's_idx'])

    # init model and optimizer
    model = LightGCN(num_nodes=num_users + num_items, embedding_dim=64, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_hr = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(int(graph_data.num_edges // batch_size + 1)):
            users, pos, neg = sample_bpr_batch(
                graph_data.edge_index, num_users, num_items, batch_size
            )
            users, pos, neg = users.to(device), pos.to(device), neg.to(device)

            embeddings = model(graph_data.edge_index.to(device))
            loss = bpr_loss(users, pos, neg, embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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