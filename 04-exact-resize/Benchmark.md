Profiling gives:

```
==9104== Profiling application: ./cuda_fun_resize aerial-photography-1080.jpg aerial-photography-1080-output.jpg
==9104== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  67.035ms         3  22.345ms  21.812ms  22.665ms  void downscale<int=2>(float*, int, int, int, int, float*)
      API calls:   66.72%  135.53ms         6  22.588ms  12.962us  135.44ms  cudaMallocManaged
                   32.99%  67.024ms         1  67.024ms  67.024ms  67.024ms  cudaDeviceSynchronize
                    0.11%  229.36us        97  2.3640us     196ns  96.920us  cuDeviceGetAttribute
                    0.11%  221.86us         1  221.86us  221.86us  221.86us  cuDeviceTotalMem
                    0.04%  72.227us         3  24.075us  4.5970us  59.218us  cudaLaunchKernel
                    0.02%  34.411us         1  34.411us  34.411us  34.411us  cuDeviceGetName
                    0.01%  11.782us         6  1.9630us     410ns  9.3040us  cudaFree
                    0.01%  10.198us         3  3.3990us     325ns  9.3910us  cudaMemPrefetchAsync
                    0.00%  4.4230us         1  4.4230us  4.4230us  4.4230us  cuDeviceGetPCIBusId
                    0.00%  2.2830us         3     761ns     209ns  1.6230us  cuDeviceGetCount
                    0.00%  1.0350us         2     517ns     250ns     785ns  cuDeviceGet
                    0.00%     389ns         1     389ns     389ns     389ns  cuDeviceGetUuid
```

Setup: 2060, linux, i7 7700.
