using Test
using LogicCircuits
using ProbabilisticCircuits
using Random: MersenneTwister
using CUDA: functional

include("../helper/gpu.jl")

function histogram_matches_likelihood(samples::Matrix{Bool}, worlds, loglikelihoods)
    hist = Dict{BitVector,Int}()
    for i = 1:size(samples,1)
        sample = BitVector(samples[i,:]) 
        hist[sample] = get(hist, sample, 0) + 1
    end
    for i = 1:size(worlds,1)
        exact_prob = exp(loglikelihoods[i])
        ex = BitVector(example(worlds,i))
        estim_prob = get(hist, ex, 0) / size(samples,1)
        @test exact_prob ≈ estim_prob atol=1e-2;
    end

end

@testset "Unconditional Sampling Test" begin

    rng = MersenneTwister(42)

    pc = zoo_psdd("little_4var.psdd");
    s, p = sample(pc;rng)
    @test pc(s...) ≈ p

    worlds = generate_data_all(num_variables(pc));

    loglikelihoods = EVI(pc, worlds)

    Nsamples = 2_0000

    samples, _ = sample(pc, Nsamples; rng)
    histogram_matches_likelihood(samples, worlds, loglikelihoods)

    if CUDA.functional()
        samples, _ = sample(pc, Nsamples; rng, gpu = true)
        samples_cpu = to_cpu(samples)
        histogram_matches_likelihood(samples_cpu, worlds, loglikelihoods)    
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
        @test sample_states[i,:,:] == convert(Matrix,data_all)
        @test sample_prs[i,:] ≈ loglikelihoods atol=1e-6
    end
    
    # same states on CPU and GPU
    cpu_gpu_agree(data_all) do d 
        sample(pc, num_samples, d)[1]
    end

    # same probabilities on CPU and GPU
    cpu_gpu_agree_approx(data_all) do d 
        sample(pc, num_samples, d)[2]
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
        pairs = collect(zip(sample_states[i,:,:], convert(Matrix,data_marg)))
        @test all(pairs) do (f,m)
            ismissing(m) || f == m
        end

        # probability does not exceed MAP probability
        @test all(sample_prs[i,:] .<= map_pr .+ 1e-6)
    end


end