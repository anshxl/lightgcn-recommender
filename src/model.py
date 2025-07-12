import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN as _LightGCN
import random

# LightGCN with dropout
class LGCNWithDropout(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int = 64, num_layers: int = 3, dropout: int = 0.1):
        super().__init__()
        self.lgcn = _LightGCN(num_nodes, embedding_dim, num_layers)
        
        # Simple embedding lookup for all nodes
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, edge_index):
        x = self.embedding.weight
        out = self.lgcn(x, edge_index)
        out = self.dropout(out)
        return out
    
# Define BPR loss function
def bpr_loss(users, pos_items, neg_items, embeddings):
    """
    users, pos_items, neg_items: 1d Tensors of same length B
    embeddings: (num_nodes, emb_dim)
    """
    u_emb = embeddings[users]
    pos_emb = embeddings[pos_items]
    neg_emb = embeddings[neg_items]

    pos_scores = (u_emb * pos_emb).sum(dim=1)  # B
    neg_scores = (u_emb * neg_emb).sum(dim=1)  # B

    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss

# BPR sampling
def sample_bpr_batch(batch_users: torch.Tensor,
                     user2items: dict,
                     num_users: int,
                     num_items: int):
    """
    batch_users: LongTensor of shape [B] with global user indices
    user2items: dict mapping each user (0..num_users-1) to list of item indices
    num_users, num_items: ints
    Returns: (users, pos_items, neg_items) LongTensors of length ≤ B
    """
    users, pos_items, neg_items = [], [], []

    for u in batch_users.tolist():
        pos_cands = user2items.get(u, [])
        if not pos_cands:
            continue

        pos_i = random.choice(pos_cands)
        # sample a neg item outside user's positives
        neg_i = random.randrange(num_users, num_users + num_items)
        while neg_i in pos_cands:
            neg_i = random.randrange(num_users, num_users + num_items)

        users.append(u)
        pos_items.append(pos_i)
        neg_items.append(neg_i)

    device = batch_users.device
    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )