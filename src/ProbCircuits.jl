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

function train_parameters(pc::ProbCircuit△, data::XBatches{Bool}, afc::AggregateFlowCircuit△=aggr_flow_circuit(pc, data);
                          pseudocount, compute_ll=false)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert afc[end].origin == pc[end] "AggregateFlowCircuit must originate in the ProbCircuit"
    collect_aggr_flows(afc, data)
    estimate_parameters(afc, pseudocount)
    train_ll = compute_ll ? log_likelihood(afc, data) : nothing
    (afc, train_ll)
end

function test_parameters(pc::ProbCircuit△, data::XBatches{Bool}, afc::AggregateFlowCircuit△=aggr_flow_circuit(pc, data))
    @assert feature_type(data) == Bool "Can only test probabilistic circuits on Bool data"
    collect_aggr_flows(afc, data)
    test_ll = log_likelihood(afc, data)
    (afc, test_ll)
end

aggr_flow_circuit(pc, data) = AggregateFlowCircuit(pc, aggr_weight_type(data))

function estimate_parameters(afc::AggregateFlowCircuit△, pseudocount)
    for n in afc
         # turns aggregate statistics into theta parameters
        estimate_parameters(n, pseudocount)
    end
end

estimate_parameters(::AggregateFlowCircuitNode, ::Any) = () # do nothing
function estimate_parameters(n::AggregateFlow⋁, pseudocount)
    origin = n.origin::Prob⋁
    if num_children(n) == 1
        origin.log_thetas .= 0.0
    else
        smoothed_aggr_flow = (n.aggr_flow + pseudocount)
        uniform_pseudocount = pseudocount / num_children(n)
        origin.log_thetas .= log.( (n.aggr_flow_children .+ uniform_pseudocount) ./ smoothed_aggr_flow )
        @assert isapprox(sum(exp.(origin.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one: $(exp.(origin.log_thetas))"
        #normalize away any leftover error
        origin.log_thetas .- log(sum(exp.(origin.log_thetas))) # TODO this can be optimized
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
log_likelihood(n::AggregateFlow⋁) = sum(n.origin.log_thetas .* n.aggr_flow_children)

function log_likelihood_per_instance(pc::ProbCircuit△, batch::PlainXData{Bool})
    opts = (flow_opts★..., el_type=Bool, compact⋁=false) #keep default options but insist on Bool flows
    fc = FlowCircuit(pc, num_examples(batch), Bool, FlowCache(), opts)
    pass_up_down(fc, plain_x_data(batch))
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