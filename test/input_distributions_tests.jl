using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
import ProbabilisticCircuits as PCs



@testset "input distributions" begin
    
    ns = input_nodes(ProbCircuit, LiteralDist, 3; sign = true)
    @test length(ns) == 3
    @test ns[1].randvar == 1
    @test ns[1].dist.sign == true
    
    ns = input_nodes(ProbCircuit, BernoulliDist, 3; p = Float32(0.5))
    @test length(ns) == 3
    @test ns[1].randvar == 1
    @test ns[1].dist.logp ≈ log(0.5)
    
    ns = input_nodes(ProbCircuit, CategoricalDist, 3; num_cats = 4)
    @test length(ns) == 3
    @test ns[1].randvar == 1
    @test all(ns[1].dist.logps .≈ [log(0.25), log(0.25), log(0.25), log(0.25)])
    
end