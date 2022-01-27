using Pkg; Pkg.activate("$(@__DIR__)")

using DataFrames, CUDA, BenchmarkTools
using LogicCircuits: batch, num_elements, num_nodes
using ProbabilisticCircuits: read_fast, ParamBitCircuit, to_gpu, isgpu,
estimate_parameters_em!, estimate_parameters_em_per_batch!, 
update_pc_params_from_pbc!, marginal_log_likelihood_avg

include("load_mnist.jl");

# device
# device!(collect(devices())[2])

# data, old code is more compatible with DataFrame
train_cpu, test_cpu = mnist_cpu()
cu_train = to_gpu(DataFrame(BitArray(train_cpu), :auto));
cu_test = to_gpu(DataFrame(BitArray(test_cpu), :auto));

batch_size = 1*512
cu_train_batched = batch(cu_train, batch_size);

# read HCLT structure
@time pc = read_fast("mnist_bits_hclt_32.jpc");
@time pbc = ParamBitCircuit(pc, cu_train_batched);
@time cu_bpc = to_gpu(pbc)

# allocate memory for MAR and flows
reuse_counts = (CuVector{Float64}(undef, num_nodes(pbc)),
                CuVector{Float64}(undef, num_elements(pbc.bitcircuit)));

reuse_v = CuMatrix{Float32}(undef, batch_size, num_nodes(pbc));
reuse_f = similar(reuse_v);


# Full batch EM
full_iters = 2
for iter = 1 : full_iters
    CUDA.@time CUDA.@sync estimate_parameters_em!(cu_bpc, cu_train_batched; 
        reuse_v, reuse_f, reuse_counts, pseudocount=0.01)
end
update_pc_params_from_pbc!(pc, cu_bpc)


# Train and test evaluation
CUDA.@time CUDA.@sync marginal_log_likelihood_avg(cu_bpc, cu_train_batched)

CUDA.@time CUDA.@sync marginal_log_likelihood_avg(cu_bpc, cu_test)


# Mini batch EM
e_start = 0.1
e_end = 0.9

mini_iters = 2
for iter = 1 : mini_iters
    e_update = e_start + (iter - 1) * (e_end - e_start) / (mini_iters - 1)
    CUDA.@time CUDA.@sync estimate_parameters_em_per_batch!(pc, cu_bpc, cu_train_batched;
        reuse_v, reuse_f, reuse_counts, pseudocount = 0.001, 
        entropy_reg=0.0, exp_update_factor = e_update)
end

# change back to matrix format
cu_train, cu_test = mnist_gpu()
nothing 