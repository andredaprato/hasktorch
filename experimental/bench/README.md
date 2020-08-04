# Benchmark results #
These benchmarks run on a completely synthetic dataset of torch tensors, with
and without automatic batching and a varying number of threads. 
## Pytorch ##
The result of the python benchmarks are 
```
No workers (unbatched):         Pytorch: 0.894s (based on 2 loops)
No workers (batch size 64):     Pytorch: 0.318s (based on 2 loops)
4 workers (batch size 64):      Pytorch: 3.069s (based on 2 loops)
``` 
and the haskell results are 

```
benchmarking dataloader with threads/100000/read tensors with 1 seed(s)
time                 171.1 ms   (116.9 ms .. 219.0 ms)
                     0.952 R²   (0.925 R² .. 0.994 R²)
mean                 140.6 ms   (127.0 ms .. 159.0 ms)
std dev              22.90 ms   (18.35 ms .. 29.08 ms)
variance introduced by outliers: 41% (moderately inflated)

benchmarking dataloader with threads/100000/read batch size (64) tensors with 1 seed(s)
time                 128.5 ms   (76.69 ms .. 170.6 ms)
                     0.864 R²   (0.562 R² .. 0.998 R²)
mean                 168.7 ms   (150.1 ms .. 195.0 ms)
std dev              33.70 ms   (21.69 ms .. 48.05 ms)
variance introduced by outliers: 55% (severely inflated)

benchmarking dataloader with threads/100000/read tensors with 4 seed(s)
time                 1.957 s    (1.736 s .. 2.293 s)
                     0.996 R²   (0.994 R² .. 1.000 R²)
mean                 1.789 s    (1.733 s .. 1.876 s)
std dev              82.76 ms   (23.00 ms .. 101.6 ms)
variance introduced by outliers: 19% (moderately inflated)

benchmarking dataloader with threads/100000/read batch size (64) tensors with 4 seed(s)
time                 775.8 ms   (544.8 ms .. 902.3 ms)
                     0.988 R²   (0.976 R² .. 1.000 R²)
mean                 751.5 ms   (669.1 ms .. 786.3 ms)
std dev              61.35 ms   (24.87 ms .. 80.51 ms)
variance introduced by outliers: 21% (moderately inflated)
```


