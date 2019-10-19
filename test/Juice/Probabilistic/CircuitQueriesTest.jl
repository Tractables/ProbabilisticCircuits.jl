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
    flow_circuit = UpFlowΔ(prob_circuit, 16, Float64, opts)
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

    flow_circuit   = FlowΔ(prob_circuit, 16, Float64, opts)
    flow_circuit_marg = FlowΔ(prob_circuit, 16, Float64, opts)


    # Comparing with down pass with fully obeserved data
    Juice.pass_up_down(flow_circuit, data_full)
    Juice.marginal_pass_up_down(flow_circuit_marg, data_full)

    for (ind, node) in enumerate(flow_circuit)
        if node isa Juice.HasDownFlow
            @test all(  isapprox.(Juice.downflow(flow_circuit[ind]), Juice.downflow(flow_circuit_marg[ind]), atol = EPS) );
        end
    end


    # Validating one example with missing features done by hand
    data_partial = XData(Int8.([-1 1 -1 1]))
    flow_circuit_part  = FlowΔ(prob_circuit, 16, Float64, opts)
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
        @test Juice.downflow(flow_circuit_part[ind_val[1]])[1] ≈ ind_val[2] atol= EPS
    end

end


#TODO this test is incorrectly named??
@testset "Sampling Test" begin
    EPS = 1e-2;
    prob_circuit = load_prob_circuit("test/circuits/little_4var.psdd");
    flow_circuit = FlowΔ(prob_circuit, 16, Bool);

    N = 4;
    data_all = XData(generate_data_all(N));

    calc_prob_all = log_likelihood_per_instance(flow_circuit, data_all);
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

@testset "Sampling With Evidence" begin
    # TODO (pashak) this test should be improved by adding few more cases
    EPS = 1e-3;
    prob_circuit = load_prob_circuit("test/circuits/little_4var.psdd");

    opts= (compact⋀=false, compact⋁=false)
    flow_circuit = UpFlowΔ(prob_circuit, 1, Float64, opts);

    N = 4;
    data = XData(Int8.([0 -1 0 -1]));
    calc_prob = marginal_log_likelihood_per_instance(flow_circuit, data);
    calc_prob = exp.(calc_prob);

    flow_circuit_all = UpFlowΔ(prob_circuit, 4, Float64, opts);
    data_all = XData(Int8.([
                            0 0 0 0;
                            0 0 0 1;
                            0 1 0 0;
                            0 1 0 1;
                        ]));
    calc_prob_all = marginal_log_likelihood_per_instance(flow_circuit_all, data_all);
    calc_prob_all = exp.(calc_prob_all);

    calc_prob_all ./= calc_prob[1]

    using DataStructures
    hist = DefaultDict{AbstractString,Float64}(0.0)

    Nsamples = 1000 * 1000
    for i = 1:Nsamples
        cur = join(Int.(sample(flow_circuit)))
        hist[cur] += 1
    end

    for k in keys(hist)
        hist[k] /= Nsamples
    end

    for ind = 1:4
        cur = join(data_all.x[ind, :])
        @test calc_prob_all[ind] ≈ hist[cur] atol= EPS;
    end
end

@testset "pr_constraint Query" begin
    # two nodes
    clt = parse_clt("./test/circuits/2.clt");
    vtree = learn_vtree_from_clt(clt; vtree_mode="balanced");
    (pc, bases) = compile_psdd_from_clt(clt, vtree);
    parents = parents_vector(pc);
    psdd = PSDDWrapper(pc, bases, parents, vtree)

    split_operation(psdd.pc[end], psdd.pc[7], Var.(1), psdd; depth = 0)
    split_operation(pc[end], pc[7], Var.(2), psdd; depth = 0)

    psdd.pc[9].log_thetas = [
        log(0.5),
        log(0.25),
        log(0.25)
    ]

    psdd.pc[5].log_thetas = [
        log(0.2),
        log(0.8)
    ]

    cache = Dict{Tuple{ProbΔNode, ProbΔNode}, Float64}()

    @test abs(pr_constraint(psdd.pc[end], psdd.pc[end], cache) - 1.0) < 1e-8
    @test abs(pr_constraint(psdd.pc[5], psdd.pc[3], cache) - 0.2) < 1e-8
    @test abs(pr_constraint(psdd.pc[5], psdd.pc[4], cache) - 0.8) < 1e-8
end
