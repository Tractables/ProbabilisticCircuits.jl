using Test
using LogicCircuits
using ProbabilisticCircuits

# This tests are supposed to test queries on the circuits
@testset "Logistic Circuit Class Conditional" begin
    # Uses a Logistic Circuit with 4 variables, and tests 3 of the configurations to 
    # match with python version.
    
    CLASSES = 2

    logistic_circuit = zoo_lc("little_4var.circuit", CLASSES)
    @test logistic_circuit isa LogisticCircuit

    # check probabilities for binary samples
    data = @. Bool([0 0 0 0; 0 1 1 0; 0 0 1 1])
    # true_weight_func = [3.43147972 4.66740416; 
    #                     4.27595352 2.83503504;
    #                     3.67415087 4.93793472]
    true_prob = [0.96867400053 0.99069084464;
                 0.98629173861 0.94453994990;
                 0.97525681816 0.99288164437]
            
    class_prob = class_conditional_likelihood_per_instance(logistic_circuit, CLASSES, data)
    for i = 1:3
        for j = 1:2
            @test true_prob[i,j] ≈ class_prob[i,j]
        end
    end

    # check probabilities for float samples
    class_prob = class_conditional_likelihood_per_instance(logistic_circuit, CLASSES, data)
    for i = 1:3
        for j = 1:2
            @test true_prob[i,j] ≈ class_prob[i,j]
        end
    end

    # check predicted_classes
    true_labels = [2, 1, 2]
    predicted_classes = predict_class(logistic_circuit, CLASSES, data)
    @test all(@. predicted_classes == true_labels)
    
    # check accuracy
    @test accuracy(logistic_circuit, CLASSES, data, true_labels) == 1.0
end