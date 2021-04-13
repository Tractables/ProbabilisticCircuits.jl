using Test
using LogicCircuits
using ProbabilisticCircuits

include("helper/plain_logic_circuits.jl")

@testset "probabilistic circuit nodes" begin

    c1 = little_3var()

    @test isdisjoint(linearize(ProbCircuit(c1)), linearize(ProbCircuit(c1)))
    
    p1 = ProbCircuit(c1)
    lit3 = children(children(p1)[1])[1]

    # traits
    @test p1 isa ProbCircuit
    @test p1 isa PlainSumNode
    @test children(p1)[1] isa PlainMulNode
    @test lit3 isa PlainProbLiteralNode
    @test issum(p1)
    @test ismul(children(p1)[1])
    @test GateType(lit3) isa LiteralGate
    @test length(mul_nodes(p1)) == 4
    
    # methods
    @test num_parameters(p1) == 10

    # extension methods
    @test literal(lit3) === literal(children(children(c1)[1])[1])
    @test variable(left_most_descendent(p1)) == Var(3)
    @test ispositive(left_most_descendent(p1))
    @test !isnegative(left_most_descendent(p1))
    @test num_nodes(p1) == 15
    @test num_edges(p1) == 18
    @test num_parameters_node(p1) == 2

    r1 = fully_factorized_circuit(ProbCircuit,10)
    @test num_parameters(r1) == 2*10+1

    @test length(mul_nodes(r1)) == 1

    # compilation tests
    @test_throws Exception compile(ProbCircuit, true)
    v1, v2, v3 = literals(ProbCircuit, 3)
    r = v1[1] * 0.3 + 0.7 * v1[2]
    @test r isa PlainSumNode
    @test all(children(r) .== [v1[1], v1[2]])
    @test all(ProbabilisticCircuits.params(r) .≈ log.([0.3, 0.7]))
    @test r * v2[1] isa PlainMulNode
    @test num_children(v1[1] * v2[1] * v3[1]) == 3
    @test num_children(v1[1] + v2[1] + v3[1]) == 3
end

@testset "probabilistic circuit transformations" begin
    prob_circ = zoo_psdd("nltcs.clt.psdd")
    split_circ = split(prob_circ, (prob_circ, prob_circ.children[1]), Var(10))
    
end