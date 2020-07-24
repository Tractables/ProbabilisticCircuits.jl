export EVI, log_likelihood_per_instance,
MAR, marginal_log_likelihood_per_instance

"""
Complete evidence queries
"""
function log_likelihood_per_instance(pc::ProbCircuit, data)
    @assert isbinarydata(data) "Can only calculate EVI on Bool data"
    
    compute_flows(origin(pc), data)
    log_likelihoods = zeros(Float64, num_examples(data))
    indices = init_array(Bool, num_examples(data))::BitVector
    
    ll(n::ProbCircuit) = ()
    ll(n::Prob⋁Node) = begin
        if num_children(n) != 1 # other nodes have no effect on likelihood
            foreach(children(origin(n)), n.log_thetas) do c, log_theta
                indices = get_downflow(origin(n), c)
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
            end
         end
    end

    foreach(ll, pc)
    log_likelihoods
end

EVI = log_proba = log_likelihood_per_instance

"""
marginal queries
"""
function marginal_log_likelihood_per_instance(pc::ProbCircuit, data)
    evaluate(pc, data)
end
MAR = marginal_log_likelihood_per_instance



