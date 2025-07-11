import pandas as pd
import torch
from torch_geometric.data import Data

# Helper function for random leave-one-out split for each user
def loo(group):
    # shuffle interactions
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(group)
    if n >= 3:
        val = group.iloc[[0]]
        test = group.iloc[[1]]
        train = group.iloc[2:]
    if n == 2:
        val = group.iloc[[0]]
        test = group.iloc[[1]]
        train = pd.DataFrame([], columns=group.columns)
    if n == 1:
        val = pd.DataFrame([], columns=group.columns)
        test = pd.DataFrame([], columns=group.columns)
        train = group
    return train, val, test

def main():
    # Load the dataset: tab-delimited TXT file with columns: user_id, song_id, play_count
    df = pd.read_csv('data/train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'playcount'])

    # Build index mapping
    unique_users = df['user'].unique()
    unique_songs = df['song'].unique()

    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    song2idx = {song: idx for idx, song in enumerate(unique_songs)}

    # Apply mapping
    df['u_idx'] = df['user'].map(user2idx)
    df['s_idx'] = df['song'].map(song2idx)

    # Split out all users
    train_list, val_list, test_list = zip(*df.groupby('user', group_keys=False).apply(loo))

    # Concatenate the results
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df   = pd.concat(val_list).reset_index(drop=True)
    test_df  = pd.concat(test_list).reset_index(drop=True)

    # Save validation and test sets
    val_df.to_csv('data/val_triplets.txt', sep='\t', index=False, header=False)
    test_df.to_csv('data/test_triplets.txt', sep='\t', index=False, header=False)

    edge_index = torch.tensor([
        train_df['u_idx'].values,
        train_df['s_idx'].values + len(unique_users)
    ], dtype=torch.long)

    # Create edge attributes
    edge_attr = torch.tensor(train_df['playcount'].values, dtype=torch.float)

    # Create the PyG data object
    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(unique_users) + len(unique_songs),
    )

    # Save the data object
    torch.save(data, "data/graph.pt")

    # Also save mappings for inference
    torch.save({
        "user2idx": user2idx,
        "idx2user": {v: k for k, v in user2idx.items()},
        "item2idx": song2idx,
        "idx2item": {v: k for k, v in song2idx.items()}
    }, "data/mappings.pt")

if __name__ == "__main__":
    main()