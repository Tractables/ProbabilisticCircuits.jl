using Test
using LogicCircuits
using ProbabilisticCircuits
using Random: MersenneTwister

@testset "Sampling Test" begin

    rng = MersenneTwister(42)

    prob_circuit = zoo_psdd("little_4var.psdd");
    data_all = generate_data_all(num_variables(prob_circuit));

    calc_prob_all = EVI(prob_circuit, data_all)

    hist = Dict{BitVector,Int}()

    Nsamples = 10_0000
    for i = 1:Nsamples
        s = sample(prob_circuit; rng)
        hist[s] = get(hist, s, 0) + 1
    end

    for i = 1:num_examples(data_all)
        exact_prob = exp(calc_prob_all[i])
        ex = BitVector(example(data_all,i))
        estim_prob = get(hist, ex, 0) / Nsamples
        @test exact_prob ≈ estim_prob atol=1e-2;
    end

end