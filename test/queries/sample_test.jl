using Test
using LogicCircuits
using ProbabilisticCircuits
using Random: MersenneTwister

@testset "Unconditional Sampling Test" begin

    rng = MersenneTwister(42)

    prob_circuit = zoo_psdd("little_4var.psdd");
    data_all = generate_data_all(num_variables(prob_circuit));

    calc_prob_all = EVI(prob_circuit, data_all)

    hist = Dict{BitVector,Int}()

    Nsamples = 2_0000
    samples, _ = sample(prob_circuit, Nsamples; rng)
    samples = map(BitVector, samples)
    for i = 1:Nsamples
        hist[samples[i]] = get(hist, samples[i], 0) + 1
    end

    for i = 1:num_examples(data_all)
        exact_prob = exp(calc_prob_all[i])
        ex = BitVector(example(data_all,i))
        estim_prob = get(hist, ex, 0) / Nsamples
        @test exact_prob ≈ estim_prob atol=1e-2;
    end
end

@testset "Conditional Sampling Test" begin

    rng = MersenneTwister(42)
    num_samples = 10

    pc = zoo_psdd("little_4var.psdd");
    data_all = generate_data_all(num_variables(pc));

    # sampling given complete data should return same data with its log likelihood 


    loglikelihoods = MAR(pc, data_all)
    sample_states, sample_prs = sample(pc, num_samples, data_all; rng)

    for i in 1:num_samples
        @test sample_states[i] == data_all
        @test sample_prs[i,:] ≈ loglikelihoods atol=1e-6
    end
    
    # sampling given partial data invariants

    data_marg = DataFrame([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])

    _, map_pr = MAP(pc, data_marg)

    sample_states, sample_prs = sample(pc, num_samples, data_marg; rng)
    
    for i in 1:num_samples

        # samples keep the partial evidence values
        @test all(zip(eachcol(sample_states[i]), eachcol(data_marg))) do (cf,cm) 
            all(zip(cf, cm)) do (f,m) 
                ismissing(m) || f == m
            end
        end

        # probability does not exceed MAP probability
        @test all(sample_prs[i,:] .<= map_pr .+ 1e-6)
    end

end