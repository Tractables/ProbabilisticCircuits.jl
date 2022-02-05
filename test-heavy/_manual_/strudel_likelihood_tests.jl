using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using Suppressor

include("../helper/gpu.jl")

@testset "Likelihood" begin
    # Uses a PC with 4 variables, and tests 3 of the configurations to
    # match with python. Also tests all probabilities sum up to 1.

    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit isa ProbCircuit;

    # Step 1. Check Probabilities for 3 samples
    data = DataFrame(BitArray([0 0 0 0; 0 1 1 0; 0 0 1 1]), :auto);
    true_prob = [0.07; 0.03; 0.13999999999999999]

    # Test Sturdel EVI
    samples, _ = sample(prob_circuit, 100000)
    mix, weights, _ = learn_strudel(DataFrame(convert(BitArray, samples), :auto); num_mix = 10,
                                    init_maxiter = 20, em_maxiter = 100, verbose = false)
    mix_calc_prob = exp.(EVI(mix, data, weights))

    @test true_prob ≈ mix_calc_prob atol = 0.1
    mix_calc_prob_all = exp.(EVI(mix, data_all))
    @test 1 ≈ sum(mix_calc_prob_all) atol = 0.1

    cpu_gpu_agree_approx(data_all) do d
        EVI(mix, d, weights)
    end
end

