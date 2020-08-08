#!/usr/bin/env python
import tensorflow as tf
import utils

dataset = tf.data.Dataset.from_tensor_slices(tf.ones(utils.DATASET_SIZE, 1))

batched_dataset = dataset.batch(64)

def read_data(dataset):
    inter = 0
    for i, batch in enumerate(batched_dataset):
        inter = batch
    return ()

if __name__ == "__main__":
    dataloaders = [batched_dataset]
    bench_labels = ["Enumerate tensorflow"]
    for (data, label) in zip(dataloaders, bench_labels):
        (runtime, loops) = utils.bench_python(read_data, data)
        utils.print_result(label, runtime, loops)


