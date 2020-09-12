using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA

function cpu_gpu_agree(f, data; atol=1e-7)
    CUDA.functional() && @test f(data) ≈ to_cpu(f(to_gpu(data))) atol=atol
end

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

    # # Validating one example with missing features done by hand
    # data_partial = Int8.([-1 1 -1 1])
    # prob_circuit = zoo_psdd("little_4var.psdd");
    # compute_exp_flows(prob_circuit, data_partial)

    # # (node index, correct down_flow_value)
    # true_vals = [(1, 0.5),
    #             (2, 1.0),
    #             (3, 0.5),
    #             (4, 0.0),
    #             (5, 0.0),
    #             (6, 0.5),
    #             (7, 0.5),
    #             (8, 0.0),
    #             (9, 1.0),
    #             (10, 1/3),
    #             (11, 1),
    #             (12, 1/3),
    #             (13, 0.0),
    #             (14, 0.0),
    #             (15, 2/3),
    #             (16, 2/3),
    #             (17, 0.0),
    #             (18, 1.0),
    #             (19, 1.0),
    #             (20, 1.0)]
    # lin = linearize(prob_circuit)
    
    # for ind_val in true_vals
    #     @test exp(get_exp_downflow(lin[ind_val[1]]; root=prob_circuit)[1]) ≈ ind_val[2] atol= EPS
    # end
end

