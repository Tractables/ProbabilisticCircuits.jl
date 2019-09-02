if endswith(@__FILE__, PROGRAM_FILE)
    # this file is run as a script
    include("../../../src/Juice/Juice.jl")
 end

using Test
using .Juice

function generate_data_all(N)
    # N = 4;
    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all, 
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    data_all
end

# This tests are supposed to test queries on the circuits
@testset "Probability of Full Evidence" begin
    # Uses a PSDD with 4 variables, and tests 3 of the configurations to 
    # match with python. Also tests all probabilities sum up to 1.

    EPS = 1e-7;
    prob_circuit = load_prob_circuit("test/circuits/little_4var.psdd");
    @test prob_circuit isa Vector{<:ProbCircuitNode};

    flow_circuit = FlowCircuit(prob_circuit, 16, Bool)
    @test flow_circuit isa Vector{<:FlowCircuitNode};

    
    # Step 1. Check Probabilities for 3 samples
    data = XData(Bool.([0 0 0 0; 0 1 1 0; 0 0 1 1]));
    true_prob = [0.07; 0.03; 0.13999999999999999]    
    
    calc_prob = log_likelihood_per_instance(flow_circuit, data)
    calc_prob = exp.(calc_prob)
    
    for i = 1:3
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end
    
    # Step 2. Add up all probabilities and see if they add up to one
    N = 4;
    # data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    # for mask = 1: (1<<N) - 1
    #     data_all = vcat(data_all, 
    #         transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
    #     );
    # end
    # data_all = XData(data_all)
    data_all = XData(generate_data_all(N))

    calc_prob_all = log_likelihood_per_instance(flow_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = EPS; 
end

@testset "Probability of partial Evidence (marginals)" begin
    EPS = 1e-7;
    prob_circuit = load_prob_circuit("test/circuits/little_4var.psdd");

    data = XData(
        Int8.([0 0 0 0; 0 1 1 0; 0 0 1 1; 
                0 0 0 -1; -1 1 0 -1; -1 -1 -1 -1; 0 -1 -1 -1])
    );
    true_prob = [0.07; 0.03; 0.13999999999999999; 
                    0.3499999999999; 0.1; 1.0; 0.8] 
    
    opts= (compact⋀=false, compact⋁=false)
    flow_circuit = FlowCircuit(prob_circuit, 16, Float64, FlowCache(), opts)
    calc_prob = marginal_log_likelihood_per_instance(flow_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:length(true_prob)
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end

end

@testset "Marginal Pass Down" begin    
    EPS = 1e-7;
    prob_circuit = load_prob_circuit("test/circuits/little_4var.psdd");
    
    N = 4
    data_full = XData(Int8.(generate_data_all(N)))
    opts= (compact⋀=false, compact⋁=false)
    
    flow_circuit   = FlowCircuit(prob_circuit, 16, Float64, FlowCache(), opts)
    flow_circuit_marg = FlowCircuit(prob_circuit, 16, Float64, FlowCache(), opts)
    

    # Comparing with down pass with fully obeserved data
    Juice.pass_up_down(flow_circuit, data_full)
    Juice.marginal_pass_up_down(flow_circuit_marg, data_full)

    for (ind, node) in enumerate(flow_circuit)
        if node isa Juice.HasPathFlow
            @test all(  isapprox.(Juice.path_flow(flow_circuit[ind]), Juice.path_flow(flow_circuit_marg[ind]), atol = EPS) );
        end
    end


    # Validating one example with missing features done by hand
    data_partial = XData(Int8.([-1 1 -1 1]))
    flow_circuit_part  = FlowCircuit(prob_circuit, 16, Float64, FlowCache(), opts)
    Juice.marginal_pass_up_down(flow_circuit_part, data_partial)

    # (node index, correct down_flow_value)
    true_vals = [(9, 0.3333333333333),
                (10, 0.0),
                (11, 0.6666666666666),
                (12, 0.0),
                (13, 1.0),
                (14, 0.5),
                (15, 0.0),
                (16, 0.5),
                (17, 0.0),
                (18, 1.0),
                (19, 1.0),
                (20, 1.0)]

    for ind_val in true_vals
        @test Juice.path_flow(flow_circuit_part[ind_val[1]])[1] ≈ ind_val[2] atol= EPS
    end

end