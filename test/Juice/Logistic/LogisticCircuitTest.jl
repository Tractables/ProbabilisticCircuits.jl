if endswith(@__FILE__, PROGRAM_FILE)
    # this file is run as a script
    include("../../../src/Juice/Juice.jl")
 end

using Test
using .Juice

# This tests are supposed to test queries on the circuits
@testset "Logistic Circuit Class Conditional" begin
    # Uses a Logistic Circuit with 4 variables, and tests 3 of the configurations to 
    # match with python version.

    EPS = 1e-7;
    my_opts = (max_factors= 2,
            compact⋀=false,
            compact⋁=false)
        
    logistic_circuit = load_logistic_circuit("test/circuits/little_4var.circuit", 2);
    @test logistic_circuit isa Vector{<:LogisticCircuitNode};

    flow_circuit = FlowCircuit(logistic_circuit, 16, Float64, my_opts)
    @test flow_circuit isa Vector{<:FlowCircuitNode};

    # Step 1. Check Probabilities for 3 samples
    data = XData(Bool.([0 0 0 0; 0 1 1 0; 0 0 1 1]));
    
    true_prob = [3.43147972 4.66740416; 
                4.27595352 2.83503504;
                3.67415087 4.93793472]
            
    CLASSES = 2
    calc_prob = class_conditional_likelihood_per_instance(flow_circuit, CLASSES, data)
    
    for i = 1:3
        for j = 1:2
            @test true_prob[i,j] ≈ calc_prob[i,j] atol= EPS;
        end
    end
end