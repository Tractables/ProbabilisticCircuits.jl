export log_prob

function log_likelihood_per_instance(pc::ProbCircuit, data::DataFrame)
end
function log_likelihood_per_instance2(pc::ProbΔ, data::XData{Bool})
    Logic.pass_up_down2(pc, data)
    log_likelihood_per_instance_cached(pc, data)
end

function log_likelihood_per_instance_cached(pc::ProbΔ, data::XData{Bool})
    log_likelihoods = zeros(num_examples(data))
    indices = init_array(Bool, num_examples(data))::BitVector
    for n in pc
         if n isa Prob⋁ && num_children(n) != 1 # other nodes have no effect on likelihood
            foreach(n.children, n.log_thetas) do c, log_theta
                indices = n.data[1] .& c.data[1]
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
            end
         end
    end
    log_likelihoods
end

# compute log likelihood
function compute_log_likelihood(pc::ProbΔ, data::XBatches{Bool})
    compute_log_likelihood(AggregateFlowΔ(pc, aggr_weight_type(data)))
end

# compute log likelihood, reusing AggregateFlowΔ but ignoring its current aggregate values
function compute_log_likelihood(afc::AggregateFlowΔ, data::XBatches{Bool})
    @assert feature_type(data) == Bool "Can only test probabilistic circuits on Bool data"
    collect_aggr_flows(afc, data)
    ll = log_likelihood(afc)
    (afc, ll)
end

# return likelihoods given current aggregate flows.
function log_likelihood(afc::AggregateFlowΔ)
    sum(n -> log_likelihood(n), afc)
end

log_likelihood(::AggregateFlowNode) = 0.0
log_likelihood(n::AggregateFlow⋁) = sum(n.origin.log_thetas .* n.aggr_flow_children)

"""
Calculates log likelihood for a batch of fully observed samples.
(Also retures the generated FlowΔ)
"""
function log_likelihood_per_instance(pc::ProbΔ, batch::PlainXData{Bool})
    fc = FlowΔ(pc, num_examples(batch), Bool)
    (fc, log_likelihood_per_instance(fc, batch))
end

function log_proba(pc::ProbΔ, batch::PlainXData{Bool})
    log_likelihood_per_instance(pc, batch)[2]
end

function log_proba(pc::ProbΔ, batch::PlainXData{Int8})
    marginal_log_likelihood_per_instance(pc, batch)[2]
end

"""
Calculate log likelihood per instance for batches of samples.
"""
function log_likelihood_per_instance(pc::ProbΔ, batches::XBatches{Bool})::Vector{Float64}
    mapreduce(b -> log_likelihood_per_instance(pc, b)[2], vcat, batches)
end

"""
Calculate log likelihood for a batch of fully observed samples.
(This is for when you already have a FlowΔ)
"""
function log_likelihood_per_instance(fc::FlowΔ, batch::PlainXData{Bool})
    @assert (prob_origin(fc[end]) isa ProbNode) "FlowΔ must originate in a ProbΔ"
    pass_up_down(fc, batch)
    log_likelihoods = zeros(num_examples(batch))
    indices = init_array(Bool, flow_length(fc))::BitVector
    for n in fc
         if n isa DownFlow⋁ && num_children(n) != 1 # other nodes have no effect on likelihood
            origin = prob_origin(n)::Prob⋁
            foreach(n.children, origin.log_thetas) do c, log_theta
                #  be careful here to allow for the Boolean multiplication to be done using & before switching to float arithmetic, or risk losing a lot of runtime!
                # log_likelihoods .+= prod_fast(downflow(n), pr_factors(c)) .* log_theta
                assign_prod(indices, downflow(n), pr_factors(c))
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
                # TODO put the lines above in Utils in order to ensure we have specialized types
            end
         end
    end
    log_likelihoods
end