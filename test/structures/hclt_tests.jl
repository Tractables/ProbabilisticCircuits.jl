using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
using CUDA
using Graphs
import ProbabilisticCircuits as PCs



@testset "hclt" begin
    
    num_vars = 3
    num_cats = 2
    
    data = cu([true true false; false true false; false false false])
    
    pc = hclt(data, 4; input_type = LiteralDist)
    _, layer = feedforward_layers(pc)
    @test pc isa ProbCircuit
    @test layer == 6
    @test num_inputs(pc) == 4
    @test randvar(pc.inputs[1].inputs[2].inputs[1]) == UInt32(1)
    @test dist(pc.inputs[1].inputs[2].inputs[1]).value == true
    @test randvar(pc.inputs[1].inputs[2].inputs[2]) == UInt32(1)
    @test dist(pc.inputs[1].inputs[2].inputs[2]).value == false

    pc = hclt(data, 4; shape=:balanced, input_type = LiteralDist)
    _, layer = feedforward_layers(pc)
    @test layer == 6
    

    # TODO FIX
    # pc = hclt(data, 4; input_type = BernoulliDist)
    
    # @test randvar(pc.inputs[1].inputs[2].inputs[1]) == UInt32(1)
    # @test pc.inputs[1].inputs[2].inputs[1].dist.logp ≈ log(0.9)
    # @test randvar(pc.inputs[1].inputs[2].inputs[2]) == UInt32(1)
    # @test pc.inputs[1].inputs[2].inputs[2].dist.logp ≈ log(0.1)

    pc = hclt(data, 4; input_type = CategoricalDist)
    
    @test randvar(pc.inputs[1].inputs[2]) == UInt32(1)
    @test pc.inputs[1].inputs[2].dist.logp ≈ log(0.5)
    
end

@testset "Balanced HCLT test" begin
    edgespair = [(1,2),(2,3),(3,4)]
    clt = PCs.clt_edges2graphs(edgespair; shape=:balanced)
    for edge in [(2,3),(2,1),(3,4)]
        @test Graphs.has_edge(clt, edge...)
    end

    edgespair = [(4,2),(2,5),(5,1),(1,6),(6,3),(3,7)]
    clt = PCs.clt_edges2graphs(edgespair; shape=:balanced)
    for edge in [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]
        @test Graphs.has_edge(clt, edge...)
    end
end