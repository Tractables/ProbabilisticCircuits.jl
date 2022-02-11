using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
import ProbabilisticCircuits as PCs

@testset "input distributions" begin

    n = input_node(ProbCircuit, LiteralDist, 1; sign = true)
    @test randvar(n) == 1

    n = input_node(ProbCircuit, BernoulliDist, 1; p = Float32(0.5))
    @test randvar(n) == 1

    n = input_node(ProbCircuit, CategoricalDist, 1; num_cats = 4)
    @test randvar(n) == 1
    
    ns = input_nodes(ProbCircuit, LiteralDist, 3; sign = true)
    @test length(ns) == 3
    @test randvar(ns[1]) == 1
    @test ns[1].dist.sign == true
    
    ns = input_nodes(ProbCircuit, BernoulliDist, 3; p = Float32(0.5))
    @test length(ns) == 3
    @test randvar(ns[1]) == 1
    @test ns[1].dist.logp ≈ log(0.5)
    
    ns = input_nodes(ProbCircuit, CategoricalDist, 3; num_cats = 4)
    @test length(ns) == 3
    @test randvar(ns[1]) == 1
    @test all(ns[1].dist.logps .≈ [log(0.25), log(0.25), log(0.25), log(0.25)])
    
end