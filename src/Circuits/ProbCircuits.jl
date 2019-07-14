#####################
# Probabilistic circuits
#####################


abstract type ProbCircuitNode<: CircuitNode end
abstract type ProbLeafNode <: ProbCircuitNode end
abstract type ProbInnerNode <: ProbCircuitNode end

struct ProbPosLeaf <: ProbLeafNode
    origin::CircuitNode
end

struct ProbNegLeaf <: ProbLeafNode
    origin::CircuitNode
end

struct Prob⋀ <: ProbInnerNode
    origin::CircuitNode
    children::Vector{<:ProbCircuitNode}
end

mutable struct Prob⋁ <: ProbInnerNode
    origin::CircuitNode
    children::Vector{<:ProbCircuitNode}
    log_thetas::Vector{Float64}
end

const ProbCircuit△ = AbstractVector{<:ProbCircuitNode}

#####################
# traits
#####################

@traitimpl Leaf{ProbLeafNode}
@traitimpl Inner{ProbInnerNode}
@traitimpl Circuit△{ProbCircuit△}

NodeType(::Type{<:ProbPosLeaf}) = PosLeaf()
NodeType(::Type{<:ProbNegLeaf}) = NegLeaf()

NodeType(::Type{<:Prob⋀}) = ⋀()
NodeType(::Type{<:Prob⋁}) = ⋁()

#####################
# constructors and conversions
#####################

const ProbCache = Dict{CircuitNode, ProbCircuitNode}

ProbCircuitNode(n::CircuitNode, cache::ProbCache) = ProbCircuitNode(NodeType(n), n, cache)

ProbCircuitNode(nt::PosLeaf, n::CircuitNode, cache::ProbCache) =
    get!(()-> ProbPosLeaf(n), cache, n)

ProbCircuitNode(nt::NegLeaf, n::CircuitNode, cache::ProbCache) =
    get!(()-> ProbNegLeaf(n), cache, n)

ProbCircuitNode(nt::⋀, n::CircuitNode, cache::ProbCache) =
    get!(cache, n) do
        Prob⋀(n, ProbCircuit(n.children, cache))
    end

ProbCircuitNode(nt::⋁, n::CircuitNode, cache::ProbCache) =
    get!(cache, n) do
        Prob⋁(n, ProbCircuit(n.children, cache), some_vector(Float64, num_children(n)))
    end

@traitfn function ProbCircuit(c::C, cache::ProbCache = ProbCache()) where {C; Circuit△{C}}
    map(n->ProbCircuitNode(n,cache), c)
end

#####################
# methods
#####################

@inline cvar(n::ProbLeafNode)::Var  = cvar(n.origin)

num_parameters(n::Prob⋁) = num_children(n)
@traitfn num_parameters(c::C) where {C; Circuit△{C}} = sum(n -> num_parameters(n), ⋁_nodes(c))

function estimate_parameters(pc::ProbCircuit△, data::XBatches{Bool}; pseudocount::Float64)
    estimate_parameters(AggregateFlowCircuit(pc, aggr_weight_type(data)), data; pseudocount=pseudocount)
end

function estimate_parameters(afc::AggregateFlowCircuit△, data::XBatches{Bool}; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (afc[end].origin isa ProbCircuitNode) "AggregateFlowCircuit must originate in a ProbCircuit"
    collect_aggr_flows(afc, data)
    estimate_parameters(afc; pseudocount=pseudocount)
    afc
end

 # turns aggregate statistics into theta parameters
function estimate_parameters(afc::AggregateFlowCircuit△; pseudocount::Float64)
    foreach(n -> estimate_parameters_node(n; pseudocount=pseudocount), afc)
end

estimate_parameters_node(::AggregateFlowCircuitNode; pseudocount::Float64) = () # do nothing
function estimate_parameters_node(n::AggregateFlow⋁; pseudocount)
    origin = n.origin::Prob⋁
    if num_children(n) == 1
        origin.log_thetas .= 0.0
    else
        smoothed_aggr_flow = (n.aggr_flow + pseudocount)
        uniform_pseudocount = pseudocount / num_children(n)
        origin.log_thetas .= log.( (n.aggr_flow_children .+ uniform_pseudocount) ./ smoothed_aggr_flow )
        @assert isapprox(sum(exp.(origin.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally: $(exp.(origin.log_thetas)), estimated from $(n.aggr_flow) and $(n.aggr_flow_children). Did you actually compute the aggregate flows?"
        #normalize away any leftover error
        origin.log_thetas .- logsumexp(origin.log_thetas)
    end
end

# compute log likelihood
function compute_log_likelihood(pc::ProbCircuit△, data::XBatches{Bool})
    compute_log_likelihood(AggregateFlowCircuit(pc, aggr_weight_type(data)))
end

# compute log likelihood, reusing AggregateFlowCircuit but ignoring its current aggregate values
function compute_log_likelihood(afc::AggregateFlowCircuit△, data::XBatches{Bool})
    @assert feature_type(data) == Bool "Can only test probabilistic circuits on Bool data"
    collect_aggr_flows(afc, data)
    ll = log_likelihood(afc)
    (afc, ll)
end

# return likelihoods given current aggregate flows.
function log_likelihood(afc::AggregateFlowCircuit△)
    sum(n -> log_likelihood(n), afc)
end

log_likelihood(::AggregateFlowCircuitNode) = 0.0
log_likelihood(n::AggregateFlow⋁) = sum(n.origin.log_thetas .* n.aggr_flow_children)

function log_likelihood_per_instance(pc::ProbCircuit△, batch::PlainXData{Bool})
    opts = (flow_opts★..., el_type=Bool, compact⋁=false) #keep default options but insist on Bool flows
    fc = FlowCircuit(pc, num_examples(batch), Bool, FlowCache(), opts)
    (fc, log_likelihood_per_instance(fc, batch))
end

function log_likelihood_per_instance(fc::FlowCircuit△, batch::PlainXData{Bool})
    @assert (fc[end].origin isa ProbCircuitNode) "FlowCircuit must originate in a ProbCircuit"
    pass_up_down(fc, batch)
    log_likelihoods = zeros(num_examples(batch))
    for n in fc
         add_log_likelihood_per_instance(n, log_likelihoods)
    end
    log_likelihoods
end

add_log_likelihood_per_instance(::FlowCircuitNode, ::Any) = () # do nothing
function add_log_likelihood_per_instance(n::Flow⋁, log_likelihoods)
    if num_children(n) != 1 # other nodes have no effect on likelihood
        origin = n.origin::Prob⋁
        foreach(n.children, origin.log_thetas) do c, theta
            log_likelihoods .+= prod_fast(π(n), pr_factors(c)) .* theta
        end
    end
end