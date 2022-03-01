using Test
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit
using Random: MersenneTwister
using CUDA


include("../helper/data.jl")
include("../helper/plain_dummy_circuits.jl")

function histogram_matches_likelihood(samples::Matrix{Bool}, worlds, loglikelihoods; EPS=1e-2)
    hist = Dict{BitVector,Int}()
    for i = 1:size(samples,1)
        sample = BitVector(samples[i,:]) 
        hist[sample] = get(hist, sample, 0) + 1
    end
    for i = 1:size(worlds,1)
        exact_prob = exp(loglikelihoods[i])
        ex = BitVector(worlds[i,:])
        estim_prob = get(hist, ex, 0) / size(samples,1)
        @test exact_prob ≈ estim_prob atol=EPS;
    end

end

@testset "Unconditional Sampling Test" begin
    rng = MersenneTwister(42)

    pc = little_4var();
    worlds = generate_data_all(num_randvars(pc));

    lls = loglikelihoods(pc, worlds; batch_size=32)

    Nsamples = 50_000
    samples = Array{Bool}(sample(pc, Nsamples, [Bool]; rng)[:,1,:])
    histogram_matches_likelihood(samples, worlds, lls)

    if CUDA.functional()
        bpc = CuBitsProbCircuit(pc)
        samples = sample(bpc, Nsamples, num_randvars(pc), [Bool]; rng)
        samples_cpu = Array{Bool}(samples[:,1,:]) # to_cpu
        histogram_matches_likelihood(samples_cpu, worlds, lls)    
    end

end

@testset "Conditional Sampling Test" begin

    rng = MersenneTwister(42)
    num_samples = 10

    pc = little_4var();
    data_all = generate_data_all(num_randvars(pc));

    if CUDA.functional()
        bpc = CuBitsProbCircuit(pc)
        data_all_gpu = cu(data_all)
    end

    # sampling given complete data should return same data with its log likelihood
    lls = loglikelihoods(pc, data_all; batch_size=16)
    sample_states = sample(pc, num_samples, data_all; batch_size=16, rng)
    for i in 1:num_samples
        @test sample_states[i,:,:] == data_all
    end
    
    if CUDA.functional()
        samples_gpu = sample(bpc, num_samples, data_all_gpu; rng)
        @test Array(samples_gpu) == sample_states
    end
    

    # sampling given partial data invariants
    data_marg = Matrix([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])
    if CUDA.functional()
        data_marg_gpu = cu(data_marg)
    end

    # Test that samples keep the partial evidence values intact
    function test_sample_match_evidence(samples_states, data_marg)
        for i in 1:num_samples
            pairs = collect(zip(sample_states[i,:,:], data_marg))
            @test all(pairs) do (f,m)
                ismissing(m) || f == m
            end
        end
    end

    # CPU
    sample_states = sample(pc, num_samples, data_marg; batch_size=8, rng)
    test_sample_match_evidence(sample_states, data_marg)

    if CUDA.functional()
        samples_gpu = sample(bpc, num_samples, data_marg_gpu; rng)
        samples_cpu = Array(samples_gpu)
        test_sample_match_evidence(samples_cpu, data_marg)
    end

    # TODO
    # Add similar test `histogram_matches_likelihood` with conditional likelihoods
    # Add sampleing for different input types
end