using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Probability of partial Evidence (marginals)" begin
    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");

    data = DataFrame([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])
    true_prob = [0.07; 0.03; 0.13999999999999999;
                    0.3499999999999; 0.1; 1.0; 0.8]

    calc_prob = MAR(prob_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:length(true_prob)
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end
end

# @testset "Marginal Pass Down" begin
#     EPS = 1e-7;
#     prob_circuit = zoo_psdd("little_4var.psdd");

#     N = 4
#     data_full = generate_data_all(N)

#     # Comparing with down pass with fully obeserved data

#     compute_flows(logic_circuit, data_full)
#     compute_exp_flows(prob_circuit, data_full)

#     lin_prob = linearize(prob_circuit)
#     lin_logic = linearize(logic_circuit)
#     for i in 1 : length(lin_prob)
#         pn = lin_prob[i]
#         ln = lin_logic[i]
#         @test all(isapprox.(exp.(get_exp_downflow(pn; root=prob_circuit)), 
#             get_downflow(ln; root=logic_circuit), atol=EPS))
#     end

#     # Validating one example with missing features done by hand
#     data_partial = Int8.([-1 1 -1 1])
#     prob_circuit = zoo_psdd("little_4var.psdd");
#     compute_exp_flows(prob_circuit, data_partial)

#     # (node index, correct down_flow_value)
#     true_vals = [(1, 0.5),
#                 (2, 1.0),
#                 (3, 0.5),
#                 (4, 0.0),
#                 (5, 0.0),
#                 (6, 0.5),
#                 (7, 0.5),
#                 (8, 0.0),
#                 (9, 1.0),
#                 (10, 1/3),
#                 (11, 1),
#                 (12, 1/3),
#                 (13, 0.0),
#                 (14, 0.0),
#                 (15, 2/3),
#                 (16, 2/3),
#                 (17, 0.0),
#                 (18, 1.0),
#                 (19, 1.0),
#                 (20, 1.0)]
#     lin = linearize(prob_circuit)
    
#     for ind_val in true_vals
#         @test exp(get_exp_downflow(lin[ind_val[1]]; root=prob_circuit)[1]) ≈ ind_val[2] atol= EPS
#     end
# end

