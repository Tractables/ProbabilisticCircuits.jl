using ProbabilisticCircuits
using DirectedAcyclicGraphs

function test_pc_equals(c1, c2)
    @test num_nodes(c1) == num_nodes(c2)
    @test num_edges(c1) == num_edges(c2)
    for (n1, n2) in zip(linearize(c1), linearize(c2))
        if issum(n1)
            @test issum(n2)
            @test all(params(n1) ≈ params(n2))
        elseif ismul(n1)
            @test ismul(n2)
        else
            @test isinput(n1) 
            @test isinput(n2)
            # TODO: might need to fix for non-literal dists
            @test params(dist(n1)) ≈ params(dist(n2))
        end
    end
end