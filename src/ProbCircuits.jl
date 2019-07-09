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
    log_theta::Vector{Float64}
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

function train_parameters(pc::ProbCircuit△, data::XBatches{Bool}, afc::AggregateFlowCircuit△=aggr_flow_circuit(pc, data); pseudocount)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert afc[end].origin == pc[end] "AggregateFlowCircuit must originate in the ProbCircuit"
    collect_aggr_flows(afc, data)
    for n in afc
         # turns aggregate statistics into theta parameters
        estimate_parameters(n, pseudocount)
    end
    train_ll = log_likelihood(afc, data)
    (pc, afc, train_ll)
end

function test_parameters(pc::ProbCircuit△, data::XBatches{Bool}, afc::AggregateFlowCircuit△=aggr_flow_circuit(pc, data))
    @assert feature_type(data) == Bool "Can only test probabilistic circuits on Bool data"
    collect_aggr_flows(afc, data)
    test_ll = log_likelihood(afc, data)
    (afc, test_ll)
end

aggr_flow_circuit(pc, data) = AggregateFlowCircuit(pc,aggr_weight_type(data))

estimate_parameters(::AggregateFlowCircuitNode, ::Any) = () # do nothing
function estimate_parameters(n::AggregateFlow⋁, pseudocount)
    origin = n.origin::Prob⋁
    if num_children(n) == 1
        origin.log_theta .= 0.0
    else
        smoothed_numerator = (n.aggr_flow + pseudocount)
        uniform_pseudocount = pseudocount / num_children(n)
        origin.log_theta .= log.( (n.aggr_flow_children .+ uniform_pseudocount) ./ smoothed_numerator )
        @assert sum(exp.(origin.log_theta)) ≈ 1.0
    end
end

# compute likelihoods of current parameters given current aggregate data stored in nodes
function log_likelihood(afc::AggregateFlowCircuit△, batches::XBatches{Bool})
    num_ex = num_examples(batches)
    num_f = num_features(batches)
    total_ll = sum(n -> log_likelihood(n), afc)
    instance_ll = total_ll/num_ex
    bits_per_pixel = -instance_ll/num_f/log(2)
    @eponymtuple(total_ll, instance_ll, bits_per_pixel)
end

log_likelihood(::AggregateFlowCircuitNode) = 0.0
log_likelihood(n::AggregateFlow⋁) = sum(n.origin.log_theta .* n.aggr_flow_children)

