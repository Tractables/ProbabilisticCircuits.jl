using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames

@testset "heuristics tests" begin
    data = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
            1 1 1 1 1 1 1 1 1 1;
            0 0 0 0 0 0 0 0 0 0;
            0 1 1 0 1 0 0 1 0 1]))
    pc1, vtree1 = learn_chow_liu_tree_circuit(data)
    for (pick_edge, pick_var) in [("eFlow","vMI"), ("eFlow", "vRand"), 
                                        ("eRand","vMI"), ("eRand", "vRand")]
        (or, and), var = heuristic_loss(pc1, data;pick_edge=pick_edge, pick_var=pick_var)
        @test or in linearize(pc1)
        @test issum(or)
        @test ismul(and) || isliteralgate(and)
        @test and in children(or)
        @test var in variables(and)
    end
end