using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
import ProbabilisticCircuits as PCs


include("helper/plain_dummy_circuits.jl")

@testset "probabilistic circuit nodes" begin

    s1 = little_3var()
    m1 = inputs(s1)[1]

    # traits
    @test s1 isa ProbCircuit
    @test s1 isa PlainSumNode
    @test m1 isa PlainMulNode
    @test issum(s1)
    @test ismul(m1)
    @test PCs.NodeType(s1) isa PCs.SumNode
    @test PCs.NodeType(m1) isa PCs.MulNode
    @test length(mulnodes(s1)) == 4
    @test length(inputnodes(s1)) == 6
    @test length(sumnodes(s1)) == 5
    
    @test num_nodes(s1) == 15
    @test num_edges(s1) == 18
    
    s1_copy = PlainProbCircuit(s1)
    @test all(isinput, intersect(linearize(s1), linearize(s1_copy))) 

    @test isinput(left_most_descendent(s1))
    @test isinput(right_most_descendent(s1))
    
    @test num_parameters_node(s1, true) == 1
    @test num_parameters_node(s1, false) == 2
    @test num_parameters(s1) == 5

    @test randvar(left_most_descendent(s1)) == randvar(left_most_descendent(s1_copy))
    @test randvar(left_most_descendent(s1)) == PCs.Var(3)
    @test dist(left_most_descendent(s1)).sign == true
    @test dist(right_most_descendent(s1)).sign == false

    lt = [PlainInputNode(i,LiteralDist(true)) for i=1:3]
    lf = [PlainInputNode(i,LiteralDist(false)) for i=1:3]

    r = lt[1] * 0.3 + 0.7 * lf[1]
    @test r isa PlainSumNode
    @test all(randvar.(inputs(r)) .== PCs.Var(1))
    @test all(params(r) .≈ log.([0.3, 0.7]))
    @test r * lt[2] isa PlainMulNode
    @test num_inputs(lt[1] * lt[2] * lt[3]) == 3
    @test num_inputs(lt[1] + lt[2] + lt[3]) == 3
    
end