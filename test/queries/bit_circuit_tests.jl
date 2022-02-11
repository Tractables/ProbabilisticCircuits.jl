using Test, ProbabilisticCircuits
using ProbabilisticCircuits: BitsInputNode

include("../helper/plain_dummy_circuits.jl")


# @testset "parameter mapping" begin
    
#     pc = little_3var()
#     bpc = bit_circuit(pc)

#     pc.params[1] = log(0.3)
#     pc.params[2] = log(0.7)

#     cache_parameters!(pc, bpc)
    
#     @test pc.params[1] ≈ log(0.5)
#     @test pc.params[2] ≈ log(0.5)
    
#     pc = little_3var_bernoulli()
#     bpc = bit_circuit(pc)
    
#     pc.inputs[1].inputs[1].dist.logp = log(0.4)
    
#     cache_parameters!(pc, bpc)
    
#     @test pc.inputs[1].inputs[1].dist.logp ≈ log(0.5)
    
#     pc = little_3var_categorical(; num_cats = UInt32(5))
#     bpc = bit_circuit(pc)
    
#     pc.inputs[1].inputs[1].dist.logps[1] = log(0.4)
    
#     cache_parameters!(pc, bpc)
    
#     @test pc.inputs[1].inputs[1].dist.logps[1] ≈ log(0.2)
    
# end