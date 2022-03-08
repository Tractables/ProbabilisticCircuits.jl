# [GPU Support](@id man-gpu)

Most queries and learning APIs support both CPU and GPU implementations. To use the GPU implementations you need to move the
circuit and the dataset to GPU, then call the corresponding API.

## Moving to GPU

### Moving Data to GPU

Currently, the APIs support `CuArray` type of gpu implemetations. One simple way to move to gpu is using the `cu` function from `CUDA.jl`.

```julia
using CUDA
train_x_gpu, test_x_gpu = cu.(train_x, test_x)
```

In case of missing values we use `Missing` type, so for example if you have categorical features with some missing values, the data type on gpu would be `CuArray{Union{Missing, UInt32}}`.

### Moving Circuit to GPU

`ProbCircuits` are stored in DAG structure and are not GPU friendly by default. So, we convert them into `BitsProbCircuits` (or bit circuits) as a lower level representation that is GPU friendly. The GPU version of bit circuits has type `CuBitsProbCircuit`, so to move your `circuit` to GPU you can simply do:

```julia
bpc = CuBitsProbCircuit(circuit);
```

## GPU APIs

The GPU supported APIs generally have the same name as their CPU counterpart, for a comprehensive list of supported functions see the API documentation. For example, we support the following on gpu:

- [`sample`](@ref)
- [`MAP`](@ref)
- [`loglikelihoods`](@ref)
- [`mini_batch_em`](@ref)
- [`full_batch_em`](@ref)