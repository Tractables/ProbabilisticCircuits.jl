using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood

include("load_mnist.jl");

# read HCLT structure
@time pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc")

@time bpc = BitsProbCircuit(pc);
@time cu_bpc = CuBitsProbCircuit(bpc);

mine=2
maxe=32
debug=false

# allocate memory for MAR and flows 
batch_size = 1*512
cu_batch = 1:batch_size;
cu_mars = CuMatrix{Float32}(undef, batch_size, length(cu_bpc.nodes));

##################################
# try single batch first
# CUDA.@time eval_circuit(cu_mars, cu_bpc, cu_train, cu_batch; mine, maxe, debug=false)
# @benchmark  CUDA.@sync eval_circuit(cu_mars, cu_bpc, cu_train, cu_batch; mine, maxe, debug)

# full epoch
CUDA.@time loglikelihood(cu_train, cu_bpc; mars_mem = cu_mars, batch_size, mine, maxe, debug)

@benchmark CUDA.@sync loglikelihood(cu_train, cu_bpc; mars_mem = cu_mars, batch_size, mine, maxe, debug)

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