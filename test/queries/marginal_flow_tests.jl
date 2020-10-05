using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA

include("../helper/gpu.jl")

@testset "Marginals" begin
    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit(false, false, false, missing) ≈ -1.0498221

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
    @test marginal_log_likelihood_avg(prob_circuit, data_marg) ≈ sum(log.(true_prob))/7
    @test marginal_all(prob_circuit, data_marg) ≈  
        [0.0 -Inf -Inf -Inf -Inf -Inf 0.0 0.0 0.0 0.0 -0.356675 -2.30259 -2.65926
        0.0 -Inf -Inf 0.0 0.0 -Inf 0.0 -Inf -Inf 0.0 -2.30259 -1.20397 -3.50656
        0.0 -Inf -Inf -Inf 0.0 0.0 0.0 0.0 -Inf -Inf -0.356675 -1.60944 -1.96611
        0.0 -Inf -Inf -Inf -Inf 0.0 0.0 0.0 0.0 0.0 -0.356675 -0.693147 -1.04982
        0.0 -Inf 0.0 0.0 -Inf 0.0  0.0 -Inf 0.0 0.0 -1.60944 -0.693147 -2.30259
        0.0 -Inf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -2.98023f-8 -7.45058f-9 -3.72529f-8
        0.0 -Inf -Inf 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.223144 -7.45058f-9 -0.223144]

    cpu_gpu_agree_approx(data_marg) do d 
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

    cpu_gpu_agree_approx(data_full) do d 
        marginal_all(prob_circuit, d)
    end

end

@testset "Marginal flows" begin
    
    prob_circuit = zoo_psdd("little_4var.psdd");

    function test_flows(data)
        # Comparing with down pass with fully observed data

        data_f = CUDA.@allowscalar Float64.(data)

        _, f1 = satisfies_flows(prob_circuit, data_f)
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
    
    cpu_gpu_agree_approx(data_full) do d 
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

