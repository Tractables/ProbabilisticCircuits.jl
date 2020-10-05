using Test
using LogicCircuits
using ProbabilisticCircuits
using LightGraphs
using MetaGraphs
using DataFrames
using Suppressor

@testset "learn chow-liu tree tests" begin
    EPS = 1e-6

    function test_chow_liu_tree(t, train_x)
        @test t isa CLT
        @test t isa MetaDiGraph
        @test is_directed(t)
        @test !has_self_loops(t)
        @test ne(t) == num_features(train_x) - 1
        @test nv(t) == num_features(train_x)
        @test_nowarn @suppress_out print_tree(t)
        for v in 1 : num_features(train_x)
            @test has_vertex(t, v)
            p = get_prop(t, v, :parent)
            @test p == 0 || has_edge(t, p, v)
            cpt = get_prop(t, v, :cpt)
            if p == 0
                @test cpt[1] ≈ sum(train_x[:,v]) / num_examples(train_x) atol=EPS
                @test cpt[1] + cpt[0] ≈ 1.0 atol=EPS
            else
                @test cpt[(1,1)] ≈ sum(train_x[:,p] .& train_x[:,v]) / sum(train_x[:,p]) atol=EPS 
                @test cpt[(1,0)] ≈ sum(.!train_x[:,p] .& train_x[:,v]) / (num_examples(train_x) - sum(train_x[:,p])) atol=EPS
                @test cpt[(1,1)] + cpt[(0, 1)] ≈ 1.0 atol=EPS
                @test cpt[(1,0)] + cpt[(0, 0)] ≈ 1.0 atol=EPS
            end
        end
        for (e1, e2) in [(1, 3), (1, 4), (1, 7), (1, 9), 
                        (2, 8), (2, 10), (3, 2), (3, 5), (4, 6)]
            @test has_edge(t, e1, e2) || has_edge(t, e2, e1)
        end

    end

    train_x = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
                                1 1 1 1 1 1 1 1 1 1;
                                0 0 0 0 0 0 0 0 0 0;
                                0 1 1 0 1 0 0 1 0 1]))
    @test_throws Exception learn_chow_liu_tree(train_x; α = 0.0, clt_root="")
    t1 = learn_chow_liu_tree(train_x; α=0.0, clt_root="graph_center")
    test_chow_liu_tree(t1, train_x)
    t2 = learn_chow_liu_tree(train_x; α=0.0, clt_root="rand")
    test_chow_liu_tree(t2, train_x)
    train_x2, _, _ = twenty_datasets("nltcs")
    t3 = learn_chow_liu_tree(train_x2; α=0.0)
    for name in LogicCircuits.LoadSave.twenty_dataset_names[1:20]
        train_x2, _, _ = twenty_datasets(name)
        t3 = learn_chow_liu_tree(train_x2; α=0.0)
    end
end