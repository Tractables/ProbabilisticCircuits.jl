using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "mixtures sharing structure tests" begin
    spc = fully_factorized_circuit(SharedProbCircuit, 10, 3)
    pc = fully_factorized_circuit(ProbCircuit, 10)
    @test spc isa SharedProbCircuit
    @test spc isa ProbCircuit
    @test issum(spc)
    @test ismul(children(spc)[1])
    @test num_components(spc) == 3
    @test num_parameters_node(spc) == num_parameters_node(pc) * 3 == 3
    @test num_parameters(spc) == num_parameters(pc) * 3 == 63
    @test num_nodes(spc) == num_nodes(pc) == 32
    @test num_edges(spc) == num_edges(pc) == 31
end