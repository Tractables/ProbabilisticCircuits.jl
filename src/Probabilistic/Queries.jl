export EVI, log_proba, log_likelihood_per_instance

using DataFrames
using LogicCircuits: UpDownFlow, UpDownFlow2

"""
Get the edge flow from logic circuit
"""
# TODO move to LogicCircuits
@inline isfactorized(n) = n.data::UpDownFlow isa UpDownFlow2
function get_edge_flow(n, c)::BitVector
    @assert !is⋁gate(c) && is⋁gate(n)
    df = copy(n.data.downflow)
    if isfactorized(c)
        return df .&= c.data.prime_flow .& c.data.sub_flow
    else
        return df .&= c.data.downflow
    end
end

"""
Complete evidence queries
"""
function log_likelihood_per_instance(pc::ProbCircuit, data::Union{DataFrame, AbstractMatrix})
    @assert isbinarydata(data) "Can only calculate EVI on Bool data"
    
    compute_flows(origin(pc), data)
    log_likelihoods = zeros(Float64, num_examples(data))
    indices = init_array(Bool, num_examples(data))::BitVector
    
    ll(n::ProbCircuit) = ()
    ll(n::Prob⋁Node) = begin
        if num_children(n) != 1 # other nodes have no effect on likelihood
            foreach(children(origin(n)), n.log_thetas) do c, log_theta
                indices = get_edge_flow(origin(n), c)
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
            end
         end
    end

    foreach(ll, pc)
    log_likelihoods
end

EVI = log_proba = log_likelihood_per_instance

"""
Complete evidence queries
"""


