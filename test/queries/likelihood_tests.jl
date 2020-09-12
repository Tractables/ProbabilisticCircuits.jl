using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame

@testset "Likelihood" begin
    # Uses a PC with 4 variables, and tests 3 of the configurations to
    # match with python. Also tests all probabilities sum up to 1.

    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit isa ProbCircuit;

    # Step 1. Check Probabilities for 3 samples
    data = DataFrame(BitArray([0 0 0 0; 0 1 1 0; 0 0 1 1]));
    true_prob = [0.07; 0.03; 0.13999999999999999]

    calc_prob = EVI(prob_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:3
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end

    # Step 2. Add up all probabilities and see if they add up to one
    N = 4;
    data_all = generate_data_all(N)

    calc_prob_all = EVI(prob_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = EPS;
end