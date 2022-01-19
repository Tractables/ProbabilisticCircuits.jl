using Pkg; Pkg.activate("$(@__DIR__)")

using MLDatasets, ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows

# load data
train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

function bitsfeatures(data_int)
    data_bits = zeros(Bool, size(data_int,1), 28*28*8)
    for ex = 1:size(data_int,1), pix = 1:size(data_int,2)
        x = data_int[ex,pix]
        for b = 0:7
            if (x & (one(UInt8) << b)) != zero(UInt8)
                data_bits[ex, (pix-1)*8+b+1] = true
            end
        end
    end
    data_bits
end

train_bits = bitsfeatures(train_int);
test_bits = bitsfeatures(train_int);
cu_train = to_gpu(train_bits);
cu_test = to_gpu(test_bits);

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
CUDA.@time probs_flows_circuit(cu_flows, cu_mars, cu_bpc, cu_train, cu_batch; mine=2, maxe=32, debug=false)

# benchmark downward pass without parameter estimation
@benchmark CUDA.@sync flows_circuit(cu_flows, cu_mars, cu_bpc, nothing, batchsize; mine=2, maxe=32, debug=false)

# allocate memory for parameter estimation
node_aggr = CuVector{Float32}(undef, size(cu_flows, 2));
edge_aggr = CuVector{Float32}(undef, length(cu_bpc.edge_layers_down.vectors));

# benchmark downward pass with edge flow aggregation
edge_aggr .= 0; CUDA.@sync flows_circuit(cu_flows, cu_mars, cu_bpc, edge_aggr, batchsize; mine=2, maxe=32, debug=false)
sum(edge_aggr[1:cu_bpc.edge_layers_down.ends[1]])

@benchmark (CUDA.@sync flows_circuit(cu_flows, cu_mars, cu_bpc, edge_aggr, batchsize; mine=2, maxe=32, debug=false)) setup=(edge_aggr .= 0)

# also works with partial batches
edge_aggr .= 0; flows_circuit(cu_flows, cu_mars, cu_bpc, edge_aggr, 178; mine=2, maxe=32, debug=false)
sum(edge_aggr[1:cu_bpc.edge_layers_down.ends[1]])

# compute separate node aggregation
node_aggr .= 0; CUDA.@time aggr_node_flows(node_aggr, edge_aggr, cu_bpc)
node_aggr[end]
@benchmark (CUDA.@sync aggr_node_flows(node_aggr, edge_aggr, cu_bpc)) setup=(node_aggr .= 0)


nothing 