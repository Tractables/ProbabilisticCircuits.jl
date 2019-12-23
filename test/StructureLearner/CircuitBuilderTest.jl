using Test: @test, @testset
using LogicCircuits
using ProbabilisticCircuits

@testset "Probabilistic circuits learner tests" begin
    data = dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1)
    train_x = train(data)
    pc = learn_probabilistic_circuit(train_x; pseudocount = 1.0, algo = "chow-liu", algo_kwargs=(α=1.0, clt_root="graph_center"))
    
    # simple test
    @test pc isa ProbΔ
    @test check_parameter_integrity(pc)
    @test num_parameters(pc) == 62 
    @test pc[26].log_thetas[1] ≈ -0.023528423773273476 atol=1.0e-7

    # all evidence sums to 1
    N = num_features(train_x);
    data_all = XData(generate_data_all(N))
    fc = FlowΔ(pc, max_batch_size(train_x), Bool, opts_accumulate_flows)
    calc_prob_all = log_likelihood_per_instance(fc, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)
    @test sum_prob_all ≈ 1 atol = 1.0e-7;
end