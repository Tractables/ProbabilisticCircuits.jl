using Test
using ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode
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
    @test num_nodes(s1) == 15
    @test num_edges(s1) == 18
    @test num_parameters_node(s1, true) == 1
    @test num_parameters_node(s1, false) == 2

    # @test all(isleaf, intersect(linearize(ProbCircuit(c1)), linearize(ProbCircuit(c1))))
    
    # p1 = ProbCircuit(c1)
    # lit3 = children(children(p1)[1])[1]
    # @test lit3 isa PlainProbLiteralNode

    
    # # methods
    # @test num_parameters(p1) == 10

    # # extension methods
    # @test literal(lit3) === literal(children(children(c1)[1])[1])
    # @test variable(left_most_descendent(p1)) == Var(3)
    # @test ispositive(left_most_descendent(p1))
    # @test !isnegative(left_most_descendent(p1))

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