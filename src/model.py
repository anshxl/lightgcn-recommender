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
def sample_bpr_batch(    
    users,       # CUDA LongTensor of user indices [B]
    rowptr,      # CPU LongTensor CSR indptr [num_nodes+1]
    col,         # CPU LongTensor CSR indices [num_edges]
    num_users,
    num_items,
    num_neg=1):

    out_u, out_pos, out_neg = [], [], []
    for u in users.tolist():
        # 1) grab this user's global neighbors
        start, end = rowptr[u].item(), rowptr[u+1].item()
        neigh = col[start:end]
        # only item‐neighbors:
        pos_tensor = neigh[neigh >= num_users]
        if pos_tensor.numel() == 0:
            continue

        # build a small Python set of local item‐indices for quick membership tests
        pos_set = set((pos_tensor - num_users).tolist())

        # now sample num_neg times
        for _ in range(num_neg):
            # pick a positive at random
            pos_local = random.choice(list(pos_set))
            pos_global = pos_local + num_users

            # rejection‐sample a negative (local index)
            neg_local = random.randrange(num_items)
            while neg_local in pos_set:
                neg_local = random.randrange(num_items)
            neg_global = neg_local + num_users

            # record one triple
            out_u.append(u)
            out_pos.append(pos_global)
            out_neg.append(neg_global)

    if not out_u:
        # no valid users in this batch
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    # stack and send back to CUDA
    return (
        torch.tensor(out_u, dtype=torch.long),
        torch.tensor(out_pos, dtype=torch.long),
        torch.tensor(out_neg, dtype=torch.long),
    )