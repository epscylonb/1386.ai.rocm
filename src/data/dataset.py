import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset

class ShardDataset(Dataset):
    """Loads a single shard entirely into memory. Good for validation."""
    def __init__(self, shard_dir, split, seq_len, use_loss_mask=False):
        self.seq_len = seq_len
        files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.npy")))
        # Load all tokens from all shards into one big array
        self.data = np.concatenate([np.load(f) for f in files])
        
    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1 # +1 for target shift
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class StreamingShardDataset(IterableDataset):
    """Streams shards one by one. Essential for training on Strix Halo."""
    def __init__(self, shard_dir, split, seq_len, use_loss_mask=False):
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len
        self.shard_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.npy")))

    def __iter__(self):
        for shard_file in self.shard_files:
            data = np.load(shard_file)
            # Calculate how many full sequences are in this shard
            num_samples = len(data) // (self.seq_len + 1)
            for i in range(num_samples):
                start = i * self.seq_len
                end = start + self.seq_len + 1
                chunk = torch.from_numpy(data[start:end].astype(np.int64))
                yield chunk[:-1], chunk[1:]