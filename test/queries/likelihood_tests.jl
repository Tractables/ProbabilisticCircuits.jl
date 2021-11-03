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

    calc_prob = EVI(prob_circuit, data)
    calc_prob = exp.(calc_prob)

    @test true_prob ≈ calc_prob atol=1e-7;

    # Step 2. Add up all probabilities and see if they add up to one
    N = 4;
    data_all = generate_data_all(N)

    calc_prob_all = EVI(prob_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = 1e-7;

    cpu_gpu_agree_approx(data_all) do d 
        EVI(prob_circuit, d)
    end
    
    # make sure log-likelihoods are -Inf when the input is not satisfied
    alltrue = multiply(pos_literals(ProbCircuit,4))
    @test all(EVI(alltrue, data) .== -Inf)
    
    cpu_gpu_agree(data) do d 
        EVI(alltrue, d)
    end

    # Strudel test commented out because too slow!
    # # Test Sturdel EVI
    # samples, _ = sample(prob_circuit, 100000)
    # mix, weights, _ = learn_strudel(DataFrame(convert(BitArray, samples), :auto); num_mix = 10,
    #                                 init_maxiter = 20, em_maxiter = 100, verbose = false)
    # mix_calc_prob = exp.(EVI(mix, data, weights))

    # @test true_prob ≈ mix_calc_prob atol = 0.1
    # mix_calc_prob_all = exp.(EVI(mix, data_all))
    # @test 1 ≈ sum(mix_calc_prob_all) atol = 0.1

    # cpu_gpu_agree_approx(data_all) do d
    #     EVI(mix, d, weights)
    # end
end

@testset "Bagging models' likelihood" begin
    dfb = DataFrame(BitMatrix([true true; true true; true true; true true]), :auto)
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    # bag_dfb = bagging_dataset(dfb; num_bags = 2, frac_examples = 1.0)
    bag_dfb = Vector{DataFrame}(undef, 2)
    bag_dfb[1] = dfb[[2, 1, 3, 4], :]
    bag_dfb[2] = dfb[[4, 3, 2, 1], :]
    
    r = compile(SharedProbCircuit, r, 2)
    
    estimate_parameters!(r, bag_dfb; pseudocount = 1.0)
    
    ll = log_likelihood_per_instance(r, dfb)
    @test ll[1] ≈ -0.2107210 atol = 1e-6
    @test ll[2] ≈ -0.2107210 atol = 1e-6
    @test ll[3] ≈ -0.2107210 atol = 1e-6
    @test ll[4] ≈ -0.2107210 atol = 1e-6
    
    @test log_likelihood(r, dfb) ≈ -0.8428841 atol = 1e-6
    
    @test log_likelihood_avg(r, dfb) ≈ -0.2107210 atol = 1e-6
end
