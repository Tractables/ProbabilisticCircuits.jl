using Pkg; Pkg.activate("$(@__DIR__)")

using MLDatasets, ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood

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

mine=2
maxe=32
debug=false

# allocate memory for MAR and flows
batchsize = 512
cu_batch = CuVector(1:batchsize);
cu_mars = CuMatrix{Float32}(undef, batchsize, length(cu_bpc.nodes));

# try single batch first
CUDA.@time eval_circuit(cu_mars, cu_bpc, cu_train, cu_batch; mine, maxe, debug)

CUDA.@time loglikelihood(cu_train, cu_bpc; mars_mem = cu_mars, mine, maxe, debug)

@benchmark loglikelihood(cu_train, cu_bpc; mars_mem = cu_mars, mine, maxe, debug)

# function tune() 
#     for i=5:16
#         local b = 2^i
#         @show b
#         mars_mem = CuMatrix{Float32}(undef, b, length(cu_bpc.nodes));
#         @btime loglikelihood($cu_train, $cu_bpc; batch_size = $b, mars_mem = $mars_mem)
#     end
# end
# tune()

nothing