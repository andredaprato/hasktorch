# Benchmark results #
The benchmarks run on a completely synthetic dataset of 5 million torch tensors, on a varying number of threads.

## Pytorch ##
The result of the python benchmarks are 
```
No workers (batch size 64):     Pytorch: 12.513s (based on 4 loops)
1 workers (batch size 64):      Pytorch: 47.240s (based on 4 loops)
2 workers (batch size 64):      Pytorch: 27.444s (based on 4 loops)
``` 
TODO: figure out what is going wrong with python in the multiprocess cases.
The haskell results are 

```
benchmarking dataloader with threads/5000000/read batch size (64) tensors with 1 seed(s)
time                 4.334 s    (4.049 s .. 4.604 s)
                     0.999 R²   (0.999 R² .. 1.000 R²)
mean                 4.233 s    (4.167 s .. 4.283 s)
std dev              67.82 ms   (34.36 ms .. 93.38 ms)
variance introduced by outliers: 19% (moderately inflated)

benchmarking dataloader with threads/5000000/read batch size (64) tensors with 2 seed(s)
time                 2.114 s    (1.680 s .. 2.888 s)
                     0.982 R²   (0.975 R² .. 1.000 R²)
mean                 2.431 s    (2.262 s .. 2.543 s)
std dev              184.6 ms   (89.56 ms .. 259.0 ms)
variance introduced by outliers: 21% (moderately inflated)
```


