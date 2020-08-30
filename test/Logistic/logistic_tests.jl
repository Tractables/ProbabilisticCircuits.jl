#TODO: reinstate

# using Test
# using LogicCircuits
# using ProbabilisticCircuits

# # This tests are supposed to test queries on the circuits
# @testset "Logistic Circuit Class Conditional" begin
#     # Uses a Logistic Circuit with 4 variables, and tests 3 of the configurations to 
#     # match with python version.

#     EPS = 1e-7;
#     logistic_circuit = zoo_lc("little_4var.circuit", 2);
#     @test logistic_circuit isa LogisticCircuit;

#     # Step 1. Check Probabilities for 3 samples
#     data = Bool.([0 0 0 0; 0 1 1 0; 0 0 1 1]);
    
#     true_prob = [3.43147972 4.66740416; 
#                 4.27595352 2.83503504;
#                 3.67415087 4.93793472]
            
#     CLASSES = 2
#     calc_prob = class_conditional_likelihood_per_instance(logistic_circuit, CLASSES, data)
    
#     for i = 1:3
#         for j = 1:2
#             @test true_prob[i,j] ≈ calc_prob[i,j] atol= EPS;
#         end
#     end
# end