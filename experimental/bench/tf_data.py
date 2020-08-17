#!/usr/bin/env python
import tensorflow as tf
import utils

dataset = tf.data.Dataset.from_tensor_slices(tf.ones(utils.DATASET_SIZE, 1)).batch(64)

dataset_ranges = tf.data.Dataset.range(0,1)
dataset_interleave = dataset_ranges.interleave(lambda x:
                                               tf.data.Dataset.from_tensor_slices(tf.ones(utils.DATASET_SIZE // 2, 1)),
                                               cycle_length=2,
                                               num_parallel_calls=2).batch(64)
if __name__ == "__main__":
    dataloaders = [dataset, dataset_interleave]
    bench_labels = ["Tensorflow sequential",
                    "Tensorflow interleaved with 2 workers"]
    for (data, label) in zip(dataloaders, bench_labels):
        (runtime, loops) = utils.bench_python(utils.read_data, data)
        utils.print_result(label, runtime, loops)
