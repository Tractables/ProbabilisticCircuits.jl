using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using Suppressor

@testset "prob circuit structure learn tests" begin
    function test_pc_property(pc, vtree1, train_x)
        @test pc isa ProbCircuit 
        @test pc isa StructProbCircuit
        @test vtree1 isa PlainVtree
        @test vtree(pc) === vtree1
        @test issmooth(pc)
        @test isdeterministic(pc)
        @test isdecomposable(pc)
        @test isstruct_decomposable(pc)
        @test num_variables(pc) == num_features(train_x) == num_variables(vtree1)
        @test check_parameter_integrity(pc)
        data_all = generate_data_all(num_features(train_x))
        prob_all = EVI(pc, data_all)
        @test logsumexp(prob_all) ≈ 0.0 atol = 1e-7
    end

    data = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
                        1 1 1 1 1 1 1 1 1 1;
                        0 0 0 0 0 0 0 0 0 0;
                        0 1 1 0 1 0 0 1 0 1]))

    @test_throws "Unknown type of strategy" learn_chow_liu_tree_circuit(data; 
        pseudocount=0.0, algo_kwargs=(α=0.0, clt_root="graph_center"), vtree_kwargs=(vtree_mode="",))

    for vtree_mode in ["balanced", "linear"]
        pc, vtree1 = learn_chow_liu_tree_circuit(data; 
            pseudocount=0.0, 
            algo_kwargs=(α=0.0, clt_root="graph_center"), 
            vtree_kwargs=(vtree_mode=vtree_mode,))
        test_pc_property(pc, vtree1, train_x)
        @test num_parameters(pc) == 48
        @test num_nodes(pc) == 73
        @test log_likelihood_avg(pc, train_x) ≈ -1.8636799873410004 atol=1e-6
    end

    @suppress_out pc3 = learn_single_model(data, maxiter=10)
    test_pc_property(pc3, vtree(pc3), train_x)
    @test num_parameters(pc3) == 59
    @test num_nodes(pc3) == 91
    @test log_likelihood_avg(pc3, train_x) ≈ -3.2257220163449736 atol=1e-6
end