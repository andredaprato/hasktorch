import time
import timeit

DATASET_SIZE= 50

def read_data(dataloader):
    for i, batch in enumerate(dataloader):
        # do _something_
        inter = batch + 1
    return inter

def bench_python(f_2, data,  loops=None):
    def f():
        return f_2(data)
    if loops is None:
        s = time.perf_counter()
        f()
        e = time.perf_counter()
        duration = e - s
        loops = max(4, int(2 / duration))  # aim for 2s
    return (timeit.timeit(f, number=loops, globals=globals()) / loops, loops)


def show_time(time_s):
    if time_s < 1e-1:
        time_us = time_s * 1e6
        return "{:.3f}us".format(time_us)
    else:
        return "{:.3f}s".format(time_s)


def print_result(bench_name, py_time, loops):
    print('{}:\tLoaded data in: {} (based on {} loops)'.format(
        bench_name, show_time(py_time),  loops))
