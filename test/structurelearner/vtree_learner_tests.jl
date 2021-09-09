using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames

@testset "prob circuit structure learn tests" begin

    function test_vtree_variable_partition(v, partition)
        if v isa PlainVtreeLeafNode
            @test variable(v) == partition
        else
            test_vtree_variable_partition(v.left, partition[1])
            test_vtree_variable_partition(v.right, partition[2])
        end
    end

    data = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
                1 1 1 1 1 1 1 1 1 1;
                0 0 0 0 0 0 0 0 0 0;
                0 1 1 0 1 0 0 1 0 1]), :auto)
    p2 = [[[6,4],[2,[10,8]]],[[5,3],[1,[9,7]]]]
    p3 = [1, [[[3,[[2,[8,10]],5]],[4,6]],[7, 9]]]

    @test_throws ErrorException learn_vtree(data; alg=nothing)
    
    for (alg, vars) in zip([:topdown, :clt], [p2, p3])
        v = learn_vtree(data; alg=alg)
        test_vtree_variable_partition(v, vars)
        @test num_variables(v) == 10
        @test num_edges(v) == 18
        @test num_nodes(v) == 19
    end

    # NOTE: there are not tests for the bottom-up vtree learner because that requires `using BlossomV` which we don't want as a dependency -- but the functionality is still available for those willing to install Blossom
end
