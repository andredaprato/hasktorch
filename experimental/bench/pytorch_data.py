#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import utils


class SyntheticDataset(Dataset):

    def __init__(self, iters, transform=None):
        self.transform = transform
        self.iters = iters
        # self.data = torch.ones(utils.DATASET_SIZE, 1)

    def __len__(self):
        return self.iters
    def __getitem__(self, ix):
        sample = torch.ones(1)
        # sample = self.data[ix]
        if self.transform:
            sample = self.transform(sample)
        return sample

def make_data(batch_size, num_workers):
    return DataLoader(SyntheticDataset(utils.DATASET_SIZE),
                      batch_size=batch_size,
                      num_workers=num_workers)


def read_data(dataloader):
    for i, batch in enumerate(dataloader):
        # do _something_
        inter = batch + 1
    return inter

if __name__ == "__main__":
    # no_workers = make_data(batch_size=None, num_workers=0)
    # four_workers = make_data(batch_size=None, num_workers=4)
    no_workers_batched = make_data(batch_size=64, num_workers=0)
    one_workers_batched = make_data(batch_size=64, num_workers=1)
    two_workers_batched = make_data(batch_size=64, num_workers=2)

    dataloaders = [no_workers_batched, one_workers_batched, two_workers_batched]
    bench_labels = ["No workers (batch size 64)", "1 workers (batch size 64)", "2 workers (batch size 64)"]
    # dataloaders = [no_workers_batched, ]
    # bench_labels = ["No workers (batch size 64)"]
    for (data, label) in zip(dataloaders, bench_labels):
        (runtime, loops) = utils.bench_python(read_data, data)
        utils.print_result(label, runtime, loops)

