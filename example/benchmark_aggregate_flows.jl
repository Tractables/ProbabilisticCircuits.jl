using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows, update_params

include("load_mnist.jl")

# read HCLT structure
@time pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc")

@time bpc = BitsProbCircuit(pc);
@time cu_bpc = CuBitsProbCircuit(bpc);

# allocate memory for MAR and flows
batchsize = 512
cu_batch = CuVector(1:batchsize);
cu_mars = CuMatrix{Float32}(undef, batchsize, length(cu_bpc.nodes));
cu_flows = similar(cu_mars);

# set up data
CUDA.@time probs_flows_circuit(cu_flows, cu_mars, nothing, cu_bpc, cu_train, cu_batch; mine=2, maxe=32, debug=false)

# benchmark downward pass without parameter estimation
@benchmark CUDA.@sync flows_circuit(cu_flows, nothing, cu_bpc, cu_mars, batchsize; mine=2, maxe=32, debug=false)

# allocate memory for parameter estimation
node_aggr = CuVector{Float32}(undef, size(cu_flows, 2));
edge_aggr = CuVector{Float32}(undef, length(cu_bpc.edge_layers_down.vectors));

# downward pass with edge flow aggregation
# also works with partial batches
edge_aggr .= 0; flows_circuit(cu_flows, edge_aggr, cu_bpc, cu_mars, 178; mine=2, maxe=32, debug=false)
sum(edge_aggr[1:cu_bpc.edge_layers_down.ends[1]])

# works with full batches
edge_aggr .= 0; CUDA.@sync flows_circuit(cu_flows, edge_aggr, cu_bpc, cu_mars, batchsize; mine=2, maxe=32, debug=false)
sum(edge_aggr[1:cu_bpc.edge_layers_down.ends[1]])

@benchmark (CUDA.@sync flows_circuit(cu_flows, edge_aggr, cu_bpc, cu_mars, batchsize; mine=2, maxe=32, debug=false)) setup=(edge_aggr .= 0)

# up + down + aggr on single batch
@benchmark (CUDA.@sync probs_flows_circuit(cu_flows, cu_mars, edge_aggr, cu_bpc, cu_train, cu_batch; mine=2, maxe=32, debug=false)) setup=(edge_aggr .= 0)

# compute separate node aggregation
node_aggr .= 0; CUDA.@time aggr_node_flows(node_aggr, cu_bpc, edge_aggr)
node_aggr[end]
@benchmark (CUDA.@sync aggr_node_flows(node_aggr, cu_bpc, edge_aggr)) setup=(node_aggr .= 0)

# actually update the parameters in the edges
CUDA.@time update_params(cu_bpc, node_aggr, edge_aggr)
@benchmark (CUDA.@sync update_params(cu_bpc, node_aggr, edge_aggr))

# set up aggregate data
CUDA.@time probs_flows_circuit(cu_flows, cu_mars, edge_aggr, cu_bpc, cu_train, cu_batch; mine=2, maxe=32, debug=false)

# actually update the parameters in the edges
CUDA.@time update_params(cu_bpc, node_aggr, edge_aggr)
@benchmark (CUDA.@sync update_params(cu_bpc, node_aggr, edge_aggr))

nothing 