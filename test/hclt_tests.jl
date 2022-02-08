using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
using CUDA
import ProbabilisticCircuits as PCs



@testset "hclt" begin
    
    num_vars = 3
    num_cats = 2
    
    data = cu([true true false; false true false; false false false])
    
    pc = hclt(data, ProbCircuit; num_hidden_cats = 4, input_type = LiteralDist)

    @test pc isa ProbCircuit
    @test num_inputs(pc) == 4
    @test pc.inputs[1].inputs[2].inputs[1].randvar == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[1].dist.sign == true
    @test pc.inputs[1].inputs[2].inputs[2].randvar == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[2].dist.sign == false
    
    pc = hclt(data, ProbCircuit; num_hidden_cats = 4, input_type = BernoulliDist)
    
    @test pc.inputs[1].inputs[2].inputs[1].randvar == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[1].dist.logp ≈ log(0.9)
    @test pc.inputs[1].inputs[2].inputs[2].randvar == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[2].dist.logp ≈ log(0.1)
    
end