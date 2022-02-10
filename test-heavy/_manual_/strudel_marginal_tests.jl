using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA
using Random
using Suppressor

include("../helper/gpu.jl")

@testset "Marginals" begin
    EPS = 1e-5

    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit(false, false, false, missing) ≈ -1.0498221

    data_marg = DataFrame([false false false false;
                      false true true false;
                      false false true true;
                      false false false missing;
                      missing true false missing;
                      missing missing missing missing;
                      false missing missing missing], :auto)
    true_prob = [0.07; 0.03; 0.13999999999999999;
                    0.3499999999999; 0.1; 1.0; 0.8]

    calc_prob = exp.(MAR(prob_circuit, data_marg))
    @test true_prob ≈ calc_prob atol=EPS
    @test marginal_log_likelihood_avg(prob_circuit, data_marg) ≈ sum(log.(true_prob))/7
    marginal_all_result = marginal_all(prob_circuit, data_marg);

    marginal_all_true_answer = [0.0 -Inf -Inf -Inf -Inf -Inf 0.0 0.0 0.0 0.0 -0.356675 -2.30259 -2.65926
        0.0 -Inf -Inf 0.0 0.0 -Inf 0.0 -Inf -Inf 0.0 -2.30259 -1.20397 -3.50656
        0.0 -Inf -Inf -Inf 0.0 0.0 0.0 0.0 -Inf -Inf -0.356675 -1.60944 -1.96611
        0.0 -Inf -Inf -Inf -Inf 0.0 0.0 0.0 0.0 0.0 -0.356675 -0.693147 -1.04982
        0.0 -Inf 0.0 0.0 -Inf 0.0  0.0 -Inf 0.0 0.0 -1.60944 -0.693147 -2.30259
        0.0 -Inf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -2.98023f-8 -7.45058f-9 -3.72529f-8
        0.0 -Inf -Inf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.223144 -7.45058f-9 -0.223144];

    for pair in zip(marginal_all_result, marginal_all_true_answer)
        @test pair[1] ≈ pair[2]  atol=EPS
    end

    cpu_gpu_agree_approx(data_marg) do d
        marginal_all(prob_circuit, d)
    end

    function test_complete_mar(circ, data, weights = nothing, atol = 1e-6)
        r1 = isnothing(weights) ? EVI(circ, data) : EVI(circ, data, weights)
        @test isgpu(data) == isgpu(r1)
        r2 = isnothing(weights) ? MAR(circ, data) : MAR(circ, data, weights)
        @test isgpu(data) == isgpu(r2)
        @test r1 ≈ r2 atol = atol
    end

    data_full = generate_data_all(num_variables(prob_circuit))

    test_complete_mar(prob_circuit, data_full)
    CUDA.functional() && test_complete_mar(prob_circuit, to_gpu(data_full))

    cpu_gpu_agree_approx(data_full) do d
        marginal_all(prob_circuit, d)
    end

    # make sure log-likelihoods are -Inf when the input is not satisfied
    data = DataFrame([false true false missing;
                      false true true false;
                      missing missing missing false], :auto)
    alltrue = multiply(pos_literals(ProbCircuit,4))
    @test all(MAR(alltrue, data) .== -Inf)

    cpu_gpu_agree(data) do d
        MAR(alltrue, d)
    end

    # Strudel Marginal Flow Test
    rng = MersenneTwister(100003); # Fix the seed
    samples, _ = sample(prob_circuit, 100000; rng)
    mix, weights, _ = learn_strudel(DataFrame(convert(BitArray, samples), :auto); num_mix = 10,
                                    init_maxiter = 20, em_maxiter = 100, verbose = false)
    mix_calc_prob = exp.(MAR(mix, data_marg, weights))
    for mix_pair in zip(true_prob, mix_calc_prob)
        @test mix_pair[1] ≈ mix_pair[2]  atol=0.1
    end

    test_complete_mar(mix, data_full, weights, 0.1)
end