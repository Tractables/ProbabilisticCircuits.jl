using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA

include("../helper/gpu.jl")

@testset "Marginals" begin
    prob_circuit = zoo_psdd("little_4var.psdd");

    data_marg = DataFrame([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])
    true_prob = [0.07; 0.03; 0.13999999999999999;
                    0.3499999999999; 0.1; 1.0; 0.8]

    calc_prob = exp.(MAR(prob_circuit, data_marg))
    @test true_prob ≈ calc_prob atol=1e-7

    cpu_gpu_agree(data_marg) do d 
        marginal_all(prob_circuit, d)
    end

    function test_complete_mar(data)
        r1 = EVI(prob_circuit, data)
        r2 = MAR(prob_circuit, data)
        @test r1 ≈ r2 atol=1e-6
    end

    data_full = generate_data_all(num_variables(prob_circuit))
    
    test_complete_mar(data_full)
    CUDA.functional() && test_complete_mar(to_gpu(data_full))

    cpu_gpu_agree(data_full) do d 
        marginal_all(prob_circuit, d)
    end

end

@testset "Marginal flows" begin
    
    prob_circuit = zoo_psdd("little_4var.psdd");

    function test_flows(data)
        # Comparing with down pass with fully observed data

        _, f1 = satisfies_flows(prob_circuit, Float64.(data))
        _, f2 = marginal_flows(prob_circuit, data)

        # note: while downward pass flows should be the same, 
        # the upward pass is *not* supposed to be the same (parameters used vs not)
        
        f1 = to_cpu(f1[:,3:end]) # ignore true and false leaf
        f2 = to_cpu(f2[:,3:end]) # ignore true and false leaf

        @test f1 ≈ exp.(f2) atol=1e-6
    end

    data_full = generate_data_all(num_variables(prob_circuit))
    
    test_flows(data_full)
    CUDA.functional() && test_flows(to_gpu(data_full))
    
    cpu_gpu_agree(data_full) do d 
        _, f = marginal_flows(prob_circuit, d)
        f[:,3:end] # ignore true and false leaf
    end

    # Validating one example with missing features done by hand
    data_partial = DataFrame([missing true missing true;])
    prob_circuit = zoo_psdd("little_4var.psdd");
    _, f = marginal_flows(prob_circuit, data_partial)
    f = exp.(f)

    @test f[end] ≈ 1.0
    @test f[end-1] ≈ 1.0
    @test f[end-2] ≈ 1.0
    @test f[end-4] ≈ 2/3
    @test f[end-5] ≈ 0.0 atol=1e-7
    @test f[end-6] ≈ 1/2
    @test f[end-7] ≈ 1.0
    @test f[end-8] ≈ 1/3
    @test f[end-9] ≈ 1
    @test f[end-10] ≈ 1/2

    # correctness on gpu by transitivy with above test

end

