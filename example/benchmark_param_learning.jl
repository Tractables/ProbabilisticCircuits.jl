using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows, update_params, full_batch_em_step, full_batch_em, mini_batch_em

include("load_mnist.jl")

# read HCLT structure
@time pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc")

@time bpc = BitsProbCircuit(pc);
@time cu_bpc = CuBitsProbCircuit(bpc);

mine=2
maxe=32
debug=false
pseudocount = 0.01f0

# allocate memory for MAR and flows
node_aggr = CuVector{Float32}(undef, length(cu_bpc.nodes));
edge_aggr = CuVector{Float32}(undef, length(cu_bpc.edge_layers_down.vectors));

batch_size = 1*512
cu_batch = CuVector(1:batch_size);
cu_mars = CuMatrix{Float32}(undef, batch_size, length(cu_bpc.nodes));
cu_flows = similar(cu_mars);

cu_bpc = CuBitsProbCircuit(bpc);

# Full batch EM
num_epochs = 1
CUDA.@time full_batch_em(cu_bpc, cu_train, num_epochs; batch_size, pseudocount,
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)


# Train and test evaluation
CUDA.@time CUDA.@sync loglikelihood(cu_train, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

CUDA.@time CUDA.@sync loglikelihood(cu_test, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

# Mini batch EM
num_minibatches = 2000
inertia = 0.98
CUDA.@time mini_batch_em(cu_bpc, cu_train, num_minibatches; batch_size, pseudocount, inertia,
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)

nothing 