#!/usr/bin/env python
import torch
import time
import timeit
from torch.utils.data import Dataset, DataLoader, IterableDataset


class SyntheticDataset(IterableDataset):

    def __init__(self, iters, transform=None):
        self.transform = transform
        self.iters = iters

    def __iter__(self):
        for i in range(self.iters):
            sample = torch.ones(100)
            if self.transform:
                sample = self.transform(sample)
            yield sample

def make_data(batch_size, num_workers):
    return DataLoader(SyntheticDataset(100000), batch_size=batch_size, num_workers=num_workers)

def read_data(dataloader):
    for i, batch in enumerate(dataloader):
        # do _something_
        inter = batch + 1


def bench_python(f_2, data,  loops=None):
    def f():
        return f_2(data)
    
    if loops is None:
        f()
        s = time.perf_counter()
        f()
        e = time.perf_counter()
        duration = e - s
        loops = max(4, int(2 / duration)) # aim for 2s
    return (timeit.timeit(f, number=loops, globals=globals()) / loops, loops)


def show_time(time_s):
    if time_s < 1e-1:
        time_us = time_s * 1e6
        return "{:.3f}us".format(time_us)
    else:
        return "{:.3f}s".format(time_s)


def print_result(bench_name, py_time, loops):
    print('{}:\tPytorch: {} (based on {} loops)'.format(
        bench_name, show_time(py_time),  loops))

if __name__ == "__main__":
    no_workers = make_data(batch_size=None, num_workers=0)
    no_workers_batched = make_data(batch_size=64, num_workers=0)
    four_workers = make_data(batch_size=None, num_workers=4)
    four_workers_batched = make_data(batch_size=64, num_workers=4)
    for i, data in enumerate([no_workers, no_workers_batched, four_workers, four_workers_batched]):
        (runtime, loops_1) = bench_python(read_data, data)
        print_result("No workers", runtime, 2)
