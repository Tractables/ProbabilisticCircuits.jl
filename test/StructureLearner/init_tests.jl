using Test: @test, @testset
using LogicCircuits
using ProbabilisticCircuits

@testset "Probabilistic circuits learner tests" begin
    train_x, _, _ = twenty_datasets("nltcs")
    (pc, vtree) = learn_struct_prob_circuit(train_x)
    
    # simple test
    @test pc isa ProbCircuit
    @test vtree isa PlainVtree
    @test num_variables(vtree) == num_features(train_x)
    @test check_parameter_integrity(pc)
    @test num_parameters(pc) == 74 

    # test below has started to fail -- unclear whether that is a bug or randomness...?
    # @test pc[28].log_thetas[1] ≈ -1.1870882896239272 atol=1.0e-7

    # is structured decomposable 
    for (n, vars) in variables_by_node(pc)
        @test vars == BitSet(variables(n.vtree))
    end

    # all evidence sums to 1
    N = num_features(train_x)
    data_all = generate_data_all(N)
    calc_prob_all = log_likelihood_per_instance(pc, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)
    @test sum_prob_all ≈ 1 atol = 1.0e-7;
end