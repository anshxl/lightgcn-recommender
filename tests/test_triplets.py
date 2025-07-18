import os
import torch

def main():
    chunk_dir = 'data/bpr_triples_chunks'
    files = sorted(os.listdir(chunk_dir))
    print(f"Found {len(files)} chunk files. First few:\n  " + "\n  ".join(files[:5]))

    # Load first chunk
    first_path = os.path.join(chunk_dir, files[0])
    data = torch.load(first_path)
    print("\nContents of", files[0])
    for key in ['users', 'pos', 'neg']:
        print(f"  {key:5s}: shape = {data[key].shape}")

    unique_users = torch.unique(data['users'])
    print(f"\nUsers in this chunk: {unique_users[:10].tolist()}… up to {unique_users[-1].item()}")
    
    # Peek at the first 5 triples
    print("\nFirst 5 triples:")
    for u, p, n in zip(data['users'][:5], data['pos'][:5], data['neg'][:5]):
        print(f"  user={int(u)}, pos={int(p)}, neg={int(n)}")

if __name__ == "__main__":
    main()
    print("\nTest completed successfully!")