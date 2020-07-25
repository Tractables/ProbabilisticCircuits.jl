using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Probability of Full Evidence" begin
    # Uses a PSDD with 4 variables, and tests 3 of the configurations to
    # match with python. Also tests all probabilities sum up to 1.

    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");
    @test prob_circuit isa ProbCircuit;

    # Step 1. Check Probabilities for 3 samples
    data = Bool.([0 0 0 0; 0 1 1 0; 0 0 1 1]);
    true_prob = [0.07; 0.03; 0.13999999999999999]

    calc_prob = log_likelihood_per_instance(prob_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:3
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end

    # Step 2. Add up all probabilities and see if they add up to one
    N = 4;
    data_all = generate_data_all(N)

    calc_prob_all = log_likelihood_per_instance(prob_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)

    @test 1 ≈ sum_prob_all atol = EPS;
end

@testset "Probability of partial Evidence (marginals)" begin
    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");

    data = Int8.([0 0 0 0; 0 1 1 0; 0 0 1 1;
                0 0 0 -1; -1 1 0 -1; -1 -1 -1 -1; 0 -1 -1 -1])
    true_prob = [0.07; 0.03; 0.13999999999999999;
                    0.3499999999999; 0.1; 1.0; 0.8]

    calc_prob = marginal_log_likelihood_per_instance(prob_circuit, data)
    calc_prob = exp.(calc_prob)

    for i = 1:length(true_prob)
        @test true_prob[i] ≈ calc_prob[i] atol= EPS;
    end
end

@testset "Marginal Pass Down" begin
    EPS = 1e-7;
    prob_circuit = zoo_psdd("little_4var.psdd");
    logic_circuit = origin(prob_circuit)

    N = 4
    data_full = Bool.(generate_data_all(N))

    # Comparing with down pass with fully obeserved data
    compute_flows(logic_circuit, data_full)
    compute_flows(prob_circuit, data_full)

    for pn in linearize(prob_circuit)
        @test all(isapprox.(exp.(get_downflow(pn; root=prob_circuit)), 
            get_downflow(pn.origin; root=logic_circuit), atol=EPS))
    end

    # Validating one example with missing features done by hand
    data_partial = Int8.([-1 1 -1 1])
    prob_circuit = zoo_psdd("little_4var.psdd");
    compute_flows(prob_circuit, data_partial)

    # (node index, correct down_flow_value)
    true_vals = [(1, 0.5),
                (2, 1.0),
                (3, 0.5),
                (4, 0.0),
                (5, 0.0),
                (6, 0.5),
                (7, 0.5),
                (8, 0.0),
                (9, 1.0),
                (10, 1/3),
                (11, 1),
                (12, 1/3),
                (13, 0.0),
                (14, 0.0),
                (15, 2/3),
                (16, 2/3),
                (17, 0.0),
                (18, 1.0),
                (19, 1.0),
                (20, 1.0)]
    lin = linearize(prob_circuit)
    
    for ind_val in true_vals
        @test exp(get_downflow(lin[ind_val[1]]; root=prob_circuit)[1]) ≈ ind_val[2] atol= EPS
    end
end

function test_mpe_brute_force(prob_circuit, evidence)
    EPS = 1e-9;
    result = MPE(prob_circuit, evidence);
    for idx = 1 : num_examples(evidence)
        marg = generate_all(evidence[idx,:])
        lls = log_likelihood_per_instance(prob_circuit, marg);
        brute_mpe = marg[argmax(lls), :]

        # Compare and validate p(result[idx]) == p(brute_mpe)
        comp_data = vcat(result[idx,:]',  brute_mpe')
        lls2 = log_likelihood_per_instance(prob_circuit, comp_data);

        @test lls2[1] ≈ lls2[2] atol= EPS
    end
end

@testset "MPE Brute Force Test Small (4 var)" begin
    prob_circuit = zoo_psdd("little_4var.psdd");
    evidence = Int8.( [-1 0 0 0;
                                0 -1 -1 0;
                                1 1 1 -1;
                                1 0 1 0;
                                -1 -1 -1 1;
                                -1 -1 -1 -1] )

    test_mpe_brute_force(prob_circuit, evidence)

end

@testset "MPE Brute Force Test Big (15 var)" begin
    N = 15
    COUNT = 10

    prob_circuit = zoo_psdd("exp-D15-N1000-C4.psdd");
    evidence = Int8.(rand( (-1,0,1), (COUNT, N)))

    test_mpe_brute_force(prob_circuit, evidence)
end

@testset "Sampling Test" begin
    EPS = 1e-2;
    prob_circuit = zoo_psdd("little_4var.psdd");

    N = 4;
    data_all = generate_data_all(N);

    calc_prob_all = log_likelihood_per_instance(prob_circuit, data_all);
    calc_prob_all = exp.(calc_prob_all);

    using DataStructures
    hist = DefaultDict{AbstractString,Float64}(0.0)

    Nsamples = 1000 * 1000
    for i = 1:Nsamples
        cur = join(Int.(sample(prob_circuit)))
        hist[cur] += 1
    end

    for k in keys(hist)
        hist[k] /= Nsamples
    end

    for k in keys(hist)
        cur = parse(Int32, k, base=2) + 1 # cause Julia arrays start at 1 :(
        @test calc_prob_all[cur] ≈ hist[k] atol= EPS;
    end


end

using DataStructures
@testset "Sampling With Evidence" begin
    # TODO (pashak) this test should be improved by adding few more cases
    EPS = 1e-2;
    prob_circuit = zoo_psdd("little_4var.psdd");

    N = 4;
    data = Int8.([0 -1 0 -1])
    calc_prob = marginal_log_likelihood_per_instance(prob_circuit, data);
    calc_prob = exp.(calc_prob);

    data_all = Int8.([0 0 0 0;
                    0 0 0 1;
                    0 1 0 0;
                    0 1 0 1;]);
    calc_prob_all = marginal_log_likelihood_per_instance(prob_circuit, data_all);
    calc_prob_all = exp.(calc_prob_all);

    calc_prob_all ./= calc_prob[1]

    hist = DefaultDict{AbstractString,Float64}(0.0)

    Nsamples = 1000 * 1000
    for i = 1:Nsamples
        cur = join(Int.(sample(prob_circuit, data)))
        hist[cur] += 1
    end

    for k in keys(hist)
        hist[k] /= Nsamples
    end

    for ind = 1:4
        cur = join(data_all[ind, :])
        @test calc_prob_all[ind] ≈ hist[cur] atol= EPS;
    end
end