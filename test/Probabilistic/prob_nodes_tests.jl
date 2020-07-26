using Test
using LogicCircuits
using ProbabilisticCircuits

include("../helper/plain_logic_circuits.jl")

@testset "probabilistic circuit nodes" begin
    c1 = little_3var()
    p1 = ProbCircuit(c1)
    lit3 = children(children(p1)[1])[1]
    @test isempty(intersect(linearize(ProbCircuit(c1)), linearize(ProbCircuit(c1))))
    
    # traits
    @test p1 isa ProbCircuit
    @test p1 isa Prob⋁Node
    @test children(p1)[1] isa Prob⋀Node
    @test lit3 isa ProbLiteralNode
    @test GateType(p1) isa ⋁Gate
    @test GateType(children(p1)[1]) isa ⋀Gate
    @test GateType(lit3) isa LiteralGate

    # methods
    @test num_parameters(p1) == 10

    # extension methods
    @test literal(lit3) === literal(children(children(c1)[1])[1])
    @test variable(left_most_descendent(p1)) == Var(3)
    @test ispositive(left_most_descendent(p1))
    @test !isnegative(left_most_descendent(p1))
    @test num_nodes(p1) == 15
    @test num_edges(p1) == 18

end