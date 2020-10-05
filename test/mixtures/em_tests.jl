using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using Suppressor

@testset "mixtures EM algorithm tests" begin
    num_mix = 3
    data = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
                                1 1 1 1 1 1 1 1 1 1;
                                0 0 0 0 0 0 0 0 0 0;
                                0 1 1 0 1 0 0 1 0 1]))
    num_v = num_features(data)

    # init
    @test_throws Exception initial_weights(data, num_mix; alg="")
    w1 = initial_weights(data, num_mix; alg="cluster")
    w2 = initial_weights(data, num_mix; alg="random")
    @test sum(w1) ≈ 1.0
    @test sum(w2) ≈ 1.0
    @test length(w1) == length(w2) == num_mix

    # em
    pc = fully_factorized_circuit(ProbCircuit, num_v)
    estimate_parameters(pc, data; pseudocount=1.0)
    spc = compile(SharedProbCircuit, pc, num_mix)
    values, flows = satisfies_flows(spc, data)
    ll0 = log_likelihood_per_instance_per_component(spc, data, values, flows)
    ll = log_likelihood_per_instance(pc, data)
    @test all(ll .≈ ll0)
    w = [0.1 0.2 0.7;]
    ll1, w = one_step_em(spc, data, values, flows, w;pseudocount=1.0)
    @test sum(ll1) ≈ sum(ll)
    ll2, w = one_step_em(spc, data, values, flows, w;pseudocount=1.0)
    ll3, w = one_step_em(spc, data, values, flows, w;pseudocount=1.0)
    @test sum(ll3) > sum(ll2)
    reset_counter(spc)
    foreach(spc) do n
        if n isa SharedSumNode
            @test all(sum(exp.(n.log_probs), dims=1) .≈ 1.0)
        end
    end
    @test_nowarn @suppress_out learn_em_model(pc, data; num_mix=1, maxiter=2)
    @test_nowarn @suppress_out learn_em_model(pc, data; num_mix=num_mix, maxiter=2)
end