using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows, update_params, full_batch_em_step, full_batch_em

include("load_mnist.jl")

# read HCLT structure
@time pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc")

@time bpc = BitsProbCircuit(pc);
@time cu_bpc = CuBitsProbCircuit(bpc);

mine=2
maxe=32
debug=false
pseudocount = 1f0

# allocate memory for MAR and flows
node_aggr = CuVector{Float32}(undef, size(cu_flows, 2));
edge_aggr = CuVector{Float32}(undef, length(cu_bpc.edge_layers_down.vectors));

batch_size = 1*512
cu_batch = CuVector(1:batch_size);
cu_mars = CuMatrix{Float32}(undef, batch_size, length(cu_bpc.nodes));
cu_flows = similar(cu_mars);

cu_bpc = CuBitsProbCircuit(bpc);

CUDA.@time full_batch_em(cu_bpc, cu_train, 1; batch_size, pseudocount,
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)

CUDA.@time full_batch_em(cu_bpc, cu_train, 5; batch_size, pseudocount,
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)

CUDA.@time CUDA.@sync loglikelihood(cu_train, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

CUDA.@time CUDA.@sync loglikelihood(cu_test, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

nothing 