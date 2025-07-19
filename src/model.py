import torch
import glob
from torch.utils.data import IterableDataset
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN as _LightGCN
from torch_geometric.loader import NeighborSampler
import random
import faiss
import numpy as np

# LightGCN with dropout
class LGCNWithDropout(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int = 64, num_layers: int = 3, dropout: int = 0.1):
        super().__init__()
        self.lgcn = _LightGCN(num_nodes, embedding_dim, num_layers)
        
        # Simple embedding lookup for all nodes
        self.embedding_dim = embedding_dim
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

# BPR Loader Class
class BPRChunkDataset(IterableDataset):
    def __init__(self, chunk_dir: str, shuffle: bool = False):
        self.files = sorted(glob.glob(f"{chunk_dir}/*.pt"))
        self.shuffle = shuffle

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)
        for path in files:
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

def compute_hits(preds: np.ndarray, true_items: np.ndarray):
    """
    Count how many true_items appear in their corresponding preds rows.

    Args:
      preds: array of shape (batch_size, k) with item indices.
      true_items: array of shape (batch_size,) with the ground-truth item.
    Returns:
      The number of hits in this batch.
    """
    # For each row i, check if true_items[i] is in preds[i]
    hits_per_user = (preds == true_items[:, None]).any(axis=1)
    return int(hits_per_user.sum())

def evaluate_faiss(
    user_emb: np.ndarray,       # [U, D] float32
    item_emb: np.ndarray,       # [I, D] float32
    val_loader,                 # yields (user_ids_local, true_item_globals)
    num_users: int,
    top_k: int = 10,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> float:
    """
    Compute HR@top_k by indexing directly into precomputed embeddings.

    user_emb:   all user vectors (numpy, float32)
    item_emb:   all item vectors (numpy, float32)
    val_loader: yields (user_ids_local, true_item_global)
    """
    # 1) Normalize if you want cosine (optional)
    # faiss.normalize_L2(user_emb)
    # faiss.normalize_L2(item_emb)

    d = item_emb.shape[1]
    # 2) Build HNSW index on items
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch       = ef_search
    index.add(item_emb)   # only ~100 MB of RAM

    hits, total = 0, 0
    for user_ids, true_items in val_loader:
        # user_ids: a CPU LongTensor of local user indices [B]
        ue = user_emb[user_ids.numpy(), :]      # [B, D] float32

        # 3) FAISS search returns local item‐indices [B, top_k]
        _, I_pred = index.search(ue, top_k)

        # 4) Convert true globals to locals, then count hits
        true_locals = (true_items.numpy() - num_users).astype(int)
        # vectorized hit‐count
        # (for each row, check if true_local is among I_pred[row])
        hits += int((I_pred == true_locals[:, None]).any(axis=1).sum())
        total += ue.shape[0]

    return hits / total if total > 0 else 0.0

def infer_embeddings(model, edge_index, num_nodes, emb_dim, 
                     batch_size=4096, num_layers=3):
    """
    Compute full [num_nodes×emb_dim] embeddings on GPU in batches.
    """
    # build inference sampler
    sampler = NeighborSampler(
        edge_index,
        sizes=[-1] * num_layers,  # full neighborhood
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # prepare output buffer
    out = torch.empty(num_nodes, emb_dim, device='cuda')

    # pull initial embeddings
    x0 = model.embedding.weight
    
    model.eval()
    with torch.no_grad():
        # 4) Stream over node batches
        for batch_size_, n_id, adjs in sampler:
            # n_id: the global node IDs in this batch
            h = x0[n_id]   # gather initial features for this batch

            # adjs is a list of (edge_index, e_id, size) tuples, one per hop
            for conv, (edge_idx, _, size) in zip(model.convs, adjs):
                h_target = h[: size[1]]               # first `size[1]` rows
                h = conv((h, h_target), edge_idx.to('cuda'))

            # write the seed‐node embeddings back to out
            out[n_id[: batch_size_]] = h_target

    return out   # [num_nodes×emb_dim] on GPU
        