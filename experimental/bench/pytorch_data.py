#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import utils
import os
import time
# os.environ["OMP_NUM_THREADS"] = "1"

class SyntheticDataset(Dataset):

    def __init__(self, iters, transform=None):
        self.transform = transform
        self.iters = iters

    def __len__(self):
        return self.iters

    def __getitem__(self, ix):
        sample = torch.ones(1)
        # sample = torch.ones(10000, 30, 5)
        time.sleep(0.01)
        if self.transform:
            sample = self.transform(sample)
        return sample

def make_data(batch_size, num_workers):
    return DataLoader(SyntheticDataset(utils.DATASET_SIZE),
                      batch_size=batch_size,
                      num_workers=num_workers)



if __name__ == "__main__":
    # torch.set_num_threads(1)
    no_workers_batched = make_data(batch_size=64, num_workers=0)
    two_workers_batched = make_data(batch_size=64, num_workers=2)

    dataloaders = [no_workers_batched, two_workers_batched]
    bench_labels = ["No workers (batch size 64)", "1 workers (batch size 64)", "2 workers (batch size 64)"]
    for (data, label) in zip(dataloaders, bench_labels):
        (runtime, loops) = utils.bench_python(utils.read_data, data)
        utils.print_result(label, runtime, loops)

