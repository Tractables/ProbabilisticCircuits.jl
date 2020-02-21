using Test
using LogicCircuits
using ProbabilisticCircuits


function construct_prob_circuit()
    circuit = load_smooth_logical_circuit(zoo_psdd_file("plants.psdd"));
    @btime pc1 = ProbΔ(circuit);
    @btime pc2 = Probabilistic.ProbΔ2(circuit[end]);
    nothing
    # 101.578 ms (736243 allocations: 46.16 MiB)
    # 47.765 ms (369254 allocations: 26.97 MiB)
end

function estimate_parameters_bm()
    data = train(dataset(twenty_datasets("plants")));
    circuit = load_smooth_logical_circuit(zoo_psdd_file("plants.psdd"));

    # construct circuits
    @btime pc = Probabilistic.ProbΔ2(circuit);
    @btime pc2 = ProbΔ(circuit);

    # estimate_parameters
    @btime Probabilistic.estimate_parameters2(pc, data; pseudocount=1.0);
    @btime estimate_parameters(pc2, convert(XBatches, data); pseudocount=1.0);

    # compute log likelihood
    @btime lls = Probabilistic.log_likelihood_per_instance_cached(pc, data)
    @btime lls2 = log_likelihood_per_instance(pc2, convert(XBatches, data))

    # for (l, l2) in zip(lls, lls2)
    #     @test isapprox(l, l2, atol=1e-9)
    # end
end

