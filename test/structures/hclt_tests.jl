using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
using CUDA
import ProbabilisticCircuits as PCs



@testset "hclt" begin
    
    num_vars = 3
    num_cats = 2
    
    data = cu([true true false; false true false; false false false])
    
    pc = hclt(data, 4; input_type = LiteralDist)

    @test pc isa ProbCircuit
    @test num_inputs(pc) == 4
    @test randvar(pc.inputs[1].inputs[2].inputs[1]) == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[1].dist.sign == true
    @test randvar(pc.inputs[1].inputs[2].inputs[2]) == UInt32(1)
    @test pc.inputs[1].inputs[2].inputs[2].dist.sign == false
    
    # TODO FIX
    # pc = hclt(data, 4; input_type = BernoulliDist)
    
    # @test randvar(pc.inputs[1].inputs[2].inputs[1]) == UInt32(1)
    # @test pc.inputs[1].inputs[2].inputs[1].dist.logp ≈ log(0.9)
    # @test randvar(pc.inputs[1].inputs[2].inputs[2]) == UInt32(1)
    # @test pc.inputs[1].inputs[2].inputs[2].dist.logp ≈ log(0.1)

    pc = hclt(data, 4; input_type = CategoricalDist)
    
    @test randvar(pc.inputs[1].inputs[2]) == UInt32(1)
    @test all(pc.inputs[1].inputs[2].dist.logps .≈ [log(0.5), log(0.5)])
    
end