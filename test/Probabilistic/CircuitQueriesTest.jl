using Test
using LogicCircuits
using ProbabilisticCircuits

# This tests are supposed to test queries on the circuits
@testset "Probability of Full Evidence" begin
    # Uses a PSDD with 4 variables, and tests 3 of the configurations to
    # match with python. Also tests all probabilities sum up to 1.

    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit isa Vector{<:ProbΔNode};

    flow_circuit = FlowΔ(prob_circuit, 16, Bool)
    @test flow_circuit isa Vector{<:FlowΔNode};


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
    data_all = XData(generate_data_all(N))

    calc_prob_all = log_likelihood_per_instance(flow_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = EPS;
end

@testset "Probability of partial Evidence (marginals)" begin
    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");

    data = XData(
        Int8.([0 0 0 0; 0 1 1 0; 0 0 1 1;
                0 0 0 -1; -1 1 0 -1; -1 -1 -1 -1; 0 -1 -1 -1])
    );
    true_prob = [0.07; 0.03; 0.13999999999999999;
                    0.3499999999999; 0.1; 1.0; 0.8]

    opts = (compact⋀=false, compact⋁=false)
    flow_circuit = UpFlowΔ(prob_circuit, 16, Float64, opts)
    calc_prob = marginal_log_likelihood_per_instance(flow_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:length(true_prob)
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end

    # Now trying the other api without instantiating a flow circuit
    fc2, calc_prob2 = marginal_log_likelihood_per_instance(prob_circuit, data)
    calc_prob2 = exp.(calc_prob2)
    for i = 1:length(true_prob)
        @test true_prob[i] ≈ calc_prob2[i] atol= EPS;
    end

end

@testset "Marginal Pass Down" begin
    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");

    N = 4
    data_full = XData(Int8.(generate_data_all(N)))
    opts= (compact⋀=false, compact⋁=false)

    flow_circuit   = FlowΔ(prob_circuit, 16, Float64, opts)
    flow_circuit_marg = FlowΔ(prob_circuit, 16, Float64, opts)


    # Comparing with down pass with fully obeserved data
    pass_up_down(flow_circuit, data_full)
    marginal_pass_up_down(flow_circuit_marg, data_full)

    for (ind, node) in enumerate(flow_circuit)
        if node isa HasDownFlow
            @test all(  isapprox.(downflow(flow_circuit[ind]), downflow(flow_circuit_marg[ind]), atol = EPS) );
        end
    end


    # Validating one example with missing features done by hand
    data_partial = XData(Int8.([-1 1 -1 1]))
    flow_circuit_part  = FlowΔ(prob_circuit, 16, Float64, opts)
    ProbabilisticCircuits.marginal_pass_up_down(flow_circuit_part, data_partial)

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
        @test downflow(flow_circuit_part[ind_val[1]])[1] ≈ ind_val[2] atol= EPS
    end

end

function test_mpe_brute_force(prob_circuit, evidence)
    EPS = 1e-9;
    result = MPE(prob_circuit, evidence);
    for idx = 1 : num_examples(evidence)
        marg = XData(generate_all(evidence.x[idx,:]));
        fc, lls = log_likelihood_per_instance(prob_circuit, marg);
        brute_mpe = marg.x[argmax(lls), :]

        # Compare and validate p(result[idx]) == p(brute_mpe)
        comp_data = XData(vcat(result[idx,:]',  brute_mpe'))
        fc2, lls2 = log_likelihood_per_instance(prob_circuit, comp_data);

        @test lls2[1] ≈ lls2[2] atol= EPS
    end
end

@testset "MPE Brute Force Test Small (4 var)" begin
    prob_circuit = zoo_psdd("little_4var.psdd");
    evidence = XData( Int8.( [-1 0 0 0;
                                0 -1 -1 0;
                                1 1 1 -1;
                                1 0 1 0;
                                -1 -1 -1 1; 
                                -1 -1 -1 -1] ))

    test_mpe_brute_force(prob_circuit, evidence)

end

@testset "MPE Brute Force Test Big (15 var)" begin
    N = 15
    COUNT = 10

    prob_circuit = zoo_psdd("exp-D15-N1000-C4.psdd");
    evidence = XData(Int8.(rand( (-1,0,1), (COUNT, N) )))

    test_mpe_brute_force(prob_circuit, evidence)
end