using Test
using LogicCircuits
using ProbabilisticCircuits
using DataStructures

@testset "Sampling Test" begin
    EPS = 1e-2;
    prob_circuit = zoo_psdd("little_4var.psdd");
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
    prob_circuit = zoo_psdd("little_4var.psdd");

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