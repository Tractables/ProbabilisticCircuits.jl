# This tests are supposed to test queries on the circuits
@testset "Probability of Full Evidence" begin
    # Uses a PSDD with 4 variables, and tests 3 of the configurations to 
    # match with python. Also tests all probabilities sum up to 1.

    EPS = 1e-7;
    prob_circuit = load_psdd_prob_circuit("test/circuits/little_4var.psdd");
    @test prob_circuit isa Vector{<:ProbCircuitNode};

    flow_circuit = FlowCircuit(prob_circuit, 16, Bool)
    @test flow_circuit isa Vector{<:FlowCircuitNode};

    
    # Step 1. Check Probabilities for 3 samples
    data = XData(Bool.([0 0 0 0; 0 1 1 0; 0 0 1 1]));
    true_prob = [0.07; 0.03; 0.13999999999999999]    
    
    pass_up(flow_circuit, data)
    calc_prob = log_likelihood_per_instance(flow_circuit, data)
    calc_prob = exp.(calc_prob)
    
    for i = 1:3
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end
    
    # Step 2. Add up all probabilities and see if they add up to one
    N = 4;
    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all, 
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    data_all = XData(data_all)

    calc_prob_all = log_likelihood_per_instance(flow_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = EPS; 
end