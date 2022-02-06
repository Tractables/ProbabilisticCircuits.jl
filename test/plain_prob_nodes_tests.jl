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

    # r1 = fully_factorized_circuit(ProbCircuit,10)
    # @test num_parameters(r1) == 2*10+1

    # @test length(mul_nodes(r1)) == 1

    # # compilation tests
    # @test_throws Exception compile(ProbCircuit, true)
    # v1, v2, v3 = literals(ProbCircuit, 3)
    # r = v1[1] * 0.3 + 0.7 * v1[2]
    # @test r isa PlainSumNode
    # @test all(children(r) .== [v1[1], v1[2]])
    # @test all(ProbabilisticCircuits.params(r) .≈ log.([0.3, 0.7]))
    # @test r * v2[1] isa PlainMulNode
    # @test num_children(v1[1] * v2[1] * v3[1]) == 3
    # @test num_children(v1[1] + v2[1] + v3[1]) == 3
end