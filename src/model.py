import torch
import glob
from torch.utils.data import IterableDataset
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
    users,      # CPU LongTensor of user indices [B]
    rowptr,     # CPU LongTensor CSR indptr [num_nodes+1]
    col,        # CPU LongTensor CSR indices [num_edges]
    num_users,
    num_items,
    num_neg=1
):
    out_u, out_pos, out_neg = [], [], []

    for u in users.tolist():
        # 1) slice out this user's global neighbors
        start, end = rowptr[u].item(), rowptr[u+1].item()
        neigh = col[start:end]                  # a small 1-D tensor

        # 2) keep only item-neighbors and shift to local [0..num_items)
        pos_tensor = neigh[neigh >= num_users] - num_users
        if pos_tensor.numel() == 0:
            continue

        # 3) for each negative we want:
        for _ in range(num_neg):
            # — sample a positive by random index into pos_tensor
            idx = torch.randint(0, pos_tensor.size(0), (1,), dtype=torch.long).item()
            pos_local = pos_tensor[idx].item()
            pos_global = pos_local + num_users

            # — sample a negative by rejection, checking via tensor compare
            neg_local = random.randrange(num_items)
            # membership test via tensor equality and any()
            while (pos_tensor == neg_local).any().item():
                neg_local = random.randrange(num_items)
            neg_global = neg_local + num_users

            # record the triple
            out_u.append(u)
            out_pos.append(pos_global)
            out_neg.append(neg_global)

    if not out_u:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    return (
        torch.tensor(out_u, dtype=torch.long),
        torch.tensor(out_pos, dtype=torch.long),
        torch.tensor(out_neg, dtype=torch.long),
    )

# BPR Loader Class
class BPRChunkDataset(IterableDataset):
    def __init__(self, chunk_dir: str, shuffle: bool = False):
        self.files = sorted(glob.glob(f"{chunk_dir}/*.pt"))
        self.shuffle = shuffle

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)
        for path in self.files:
            data = torch.load(path)
            users, pos, neg = data['users'], data['pos'], data['neg']
            if self.shuffle:
                idxs = list(range(len(users)))
                random.shuffle(idxs)
            else:
                idxs = range(len(users))
            # yield triple-by-triple
            for i in idxs:
                yield users[i].item(), pos[i].item(), neg[i].item()

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