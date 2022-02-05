using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using CUDA


@testset "HCLT construction" begin
    num_vars = 3
    num_cats = 2
    
    data = DataFrame([true true false; false true false; false false false], :auto)
    
    pc = hclt(num_vars, num_cats; data, num_hidden_cats = 4)

    @test pc isa ProbCircuit
    @test num_children(pc) == 4
    @test pc.children[1].children[2].children[1].literal == Lit(1)
    @test pc.children[1].children[2].children[2].literal == Lit(-1)
end