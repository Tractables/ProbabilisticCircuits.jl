using Test, ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit

include("helper/plain_dummy_circuits.jl")

@testset "BitsPrbabilisticCircuit construction" begin
    
    pc = little_3var()
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 6
    @test length(bpc.nodes) == 11
    @test length(bpc.heap) == 0
    if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 6
        @test length(cbpc.nodes) == 11
        @test length(cbpc.heap) == 0
    end

    pc = little_3var_bernoulli()
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 3
    @test length(bpc.nodes) == 5
    @test length(bpc.heap) == 0
    if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 3
        @test length(cbpc.nodes) == 5
        @test length(cbpc.heap) == 0
    end
    
    pc = little_3var_categorical(; num_cats = 5)
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 3
    @test length(bpc.nodes) == 5
    @test length(bpc.heap) == 5*2*3
    if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 3
        @test length(cbpc.nodes) == 5
        @test length(cbpc.heap) == 5*2*3
    end
end

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