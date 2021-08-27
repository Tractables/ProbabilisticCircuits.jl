using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using Random
using Suppressor
using CUDA

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
                        0 1 1 0 1 0 0 1 0 1]), :auto)

    @test_throws "Unknown type of strategy" learn_chow_liu_tree_circuit(data; 
        pseudocount=0.0, algo_kwargs=(α=0.0, clt_root="graph_center"), vtree_kwargs=(vtree_mode="",))
    clt = learn_chow_liu_tree(data)
    @test_throws String lc = compile_sdd_from_clt(clt, Vtree(10,:balanced))
    for vtree_mode in ["balanced", "linear"]
        pc, vtree1 = learn_chow_liu_tree_circuit(data; 
            pseudocount=0.0, 
            algo_kwargs=(α=0.0, clt_root="graph_center"), 
            vtree_kwargs=(vtree_mode=vtree_mode,))
        test_pc_property(pc, vtree1, data)
        @test num_parameters(pc) == 48
        @test num_nodes(pc) == 73
        @test log_likelihood_avg(pc, data) ≈ -1.8636799873410004 atol=1e-6
    end

    pc3 = learn_circuit(data; maxiter=10, verbose = false)
    test_pc_property(pc3, vtree(pc3), data)
    @test num_parameters(pc3) == 60
    @test num_nodes(pc3) == 88
    @test log_likelihood_avg(pc3, data) ≈ -3.0466585640216746 atol=1e-6

    # Test when there are more iterations than candidates.
    data = DataFrame(convert(BitArray, rand(Bool, 100, 4)), :auto)
    @test_nowarn pc = learn_circuit(data; maxiter = 100, verbose = false)
end


@testset "learn from missing data tests" begin
    # Test for learning from missing data
    Random.seed!(10007) # Fix Seed for the test
    data = DataFrame(convert(BitArray, rand(Bool, 200, 15)), :auto)
    data_miss = make_missing_mcar(data; keep_prob=0.9)

    @test_broken pc_miss = learn_circuit_miss(data_miss; maxiter=30, verbose=false)

    if CUDA.functional()
        data_miss_gpu = to_gpu(data_miss)
        @test_broken pc_miss_gpu = learn_circuit_miss(data_miss; maxiter=30, verbose=false)
    end
end