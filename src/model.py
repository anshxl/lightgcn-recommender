import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
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
def sample_bpr_batch(edge_index, num_users, num_items, batch_size):
    """
    edge_index: [2,E] with users in [0, U), items in [U, U+I)
    returns: users, pos_items, neg_items as Tensors
    """
    users = []
    pos = []
    neg = []
    for _ in range(batch_size):
        u = random.randrange(num_users)
        # find positive item neighbors
        pos_candidates = edge_index[1, edge_index[0] == u].tolist()
        if not pos_candidates:
            continue
        i = random.choice(pos_candidates)
        # sample negative item
        j = random.randrange(num_items) + num_users
        while j in pos_candidates:
            j = random.randrange(num_items) + num_users

        users.append(u)
        pos.append(i)
        neg.append(j)

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos, dtype=torch.long),
        torch.tensor(neg, dtype=torch.long)
    )
