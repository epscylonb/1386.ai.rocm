import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import random

class ShardDataset(Dataset):
    def __init__(self, shard_dir, split, seq_len, use_loss_mask=False):
        self.seq_len = seq_len
        # Look for .bin files instead of .npy
        self.files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.bin")))
        
        if not self.files:
            raise FileNotFoundError(f"No .bin shards found for {split} in {shard_dir}")

        print(f"Loading {len(self.files)} shards for {split}...")
        # Use fromfile for raw binary shards. 
        # Assuming uint16 (standard for 32k-64k vocab tokens). 
        # If it throws an error, try np.int32.
        self.data = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in self.files])
        
    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]

class StreamingShardDataset(IterableDataset):
    def __init__(self, shard_dir, split, seq_len, use_loss_mask=False):
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len
        self.shard_files = glob.glob(os.path.join(shard_dir, f"{split}_*.bin"))

    def __iter__(self):
        # 1. Shuffle shard order every time we start a new epoch
        shuffled_shards = self.shard_files[:]
        random.shuffle(shuffled_shards)

        for shard_file in shuffled_shards:
            data = np.fromfile(shard_file, dtype=np.uint16)
            num_samples = len(data) // (self.seq_len + 1)
            
            # 2. Create a list of all start indices in this shard
            indices = list(range(num_samples))
            # 3. Shuffle the indices so we don't read the shard linearly
            random.shuffle(indices)

            for i in indices:
                start = i * self.seq_len
                end = start + self.seq_len + 1
                chunk = torch.from_numpy(data[start:end].astype(np.int64))
                yield chunk[:-1], chunk[1:]
