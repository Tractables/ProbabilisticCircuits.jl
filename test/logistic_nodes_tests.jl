using Test
using LogicCircuits
using ProbabilisticCircuits

include("helper/plain_logic_circuits.jl")

@testset "probabilistic circuit nodes" begin

    c1 = little_3var()
    classes = 2
    @test all(isleaf, intersect(linearize(LogisticCircuit(c1, classes)), linearize(LogisticCircuit(c1, classes))))
    p1 = LogisticCircuit(c1, classes)
    lit3 = children(children(p1)[1])[1]

    # traits
    @test p1 isa LogisticCircuit
    @test p1 isa LogisticInnerNode
    @test p1 isa Logistic⋁Node
    @test children(p1)[1] isa Logistic⋀Node
    @test lit3 isa LogisticLiteralNode
    @test is⋁gate(p1)
    @test is⋀gate(children(p1)[1])
    @test isliteralgate(lit3)
    @test length(or_nodes(p1)) == 5
    
    # extension methods
    @test literal(lit3) === literal(children(children(c1)[1])[1])
    @test variable(left_most_descendent(p1)) == Var(3)
    @test ispositive(left_most_descendent(p1))
    @test !isnegative(left_most_descendent(p1))
    @test num_nodes(p1) == 15
    @test num_edges(p1) == 18
    @test num_parameters(p1) == 10*2
    @test num_parameters_per_class(p1) == 10

    r1 = fully_factorized_circuit(LogisticCircuit,10; classes=classes)
    @test num_parameters(r1) == 2 * (2*10+1)
    @test length(or_nodes(r1)) == 11

end