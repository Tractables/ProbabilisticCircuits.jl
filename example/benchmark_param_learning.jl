using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows, update_params, full_batch_em_step, full_batch_em, mini_batch_em, soften

include("load_mnist.jl");

train_cpu, test_cpu = mnist_cpu()
train_gpu, test_gpu = mnist_gpu()

"Turn binary data into floating point data close to 0 and 1."
function soften_data(data; softness, pseudocount=1)
    data_marginals = ((sum(data; dims=1) .+ Float32(pseudocount/2)) 
                        ./ Float32(size(data, 1) + pseudocount))
    Float32(1-softness) * data .+ Float32(softness) * data_marginals
end

@time train_soft_gpu = soften_data(train_gpu);
train = train_soft_gpu;
train = train_gpu;

# read HCLT structure
@time pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc")

@time bpc = BitsProbCircuit(pc);
@time cu_bpc = CuBitsProbCircuit(bpc);

mine=2
maxe=32
debug=false

# allocate memory for MAR and flows
node_aggr = CuVector{Float32}(undef, length(cu_bpc.nodes));
edge_aggr = CuVector{Float32}(undef, length(cu_bpc.edge_layers_down.vectors));

batch_size = 1*512
cu_batch = CuVector(1:batch_size);
cu_mars = CuMatrix{Float32}(undef, batch_size, length(cu_bpc.nodes));
cu_flows = similar(cu_mars);

cu_bpc = CuBitsProbCircuit(bpc);

# Full batch EM
CUDA.@time full_batch_em(cu_bpc, train, 5; batch_size, pseudocount = 0.01f0, 
    softness = 0.001,
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)

# Train and test evaluation
CUDA.@time CUDA.@sync loglikelihood(train, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

CUDA.@time CUDA.@sync loglikelihood(test_gpu, cu_bpc; batch_size, mars_mem = cu_mars, mine, maxe, debug)

# Mini batch EM

CUDA.@time mini_batch_em(cu_bpc, train, 5; batch_size, 
    pseudocount = 0.001f0, softness = 0.001,param_inertia = 0.98, flow_memory=0,  
    mars_mem = cu_mars, flows_mem = cu_flows, node_aggr_mem = node_aggr, edge_aggr_mem=edge_aggr,
    mine, maxe, debug)


nothing 