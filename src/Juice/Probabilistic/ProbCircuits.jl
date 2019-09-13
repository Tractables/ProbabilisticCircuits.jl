#####################
# Probabilistic circuits
#####################

abstract type ProbCircuitNode <: DecoratorCircuitNode end
abstract type ProbLeafNode <: ProbCircuitNode end
abstract type ProbInnerNode <: ProbCircuitNode end

struct ProbLiteral <: ProbLeafNode
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

import ..Logical.NodeType # make available for extension

NodeType(::Type{<:ProbLiteral}) = LiteralLeaf()
NodeType(::Type{<:Prob⋀}) = ⋀()
NodeType(::Type{<:Prob⋁}) = ⋁()

#####################
# constructors and conversions
#####################

const ProbCache = Dict{CircuitNode, ProbCircuitNode}

ProbCircuitNode(n::CircuitNode, cache::ProbCache) = ProbCircuitNode(NodeType(n), n, cache)

ProbCircuitNode(::LiteralLeaf, n::CircuitNode, cache::ProbCache) =
    get!(()-> ProbLiteral(n), cache, n)

ProbCircuitNode(::ConstantLeaf, ::CircuitNode, ::ProbCache) =
    error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")

ProbCircuitNode(::⋀, n::CircuitNode, cache::ProbCache) =
    get!(cache, n) do
        Prob⋀(n, ProbCircuit(n.children, cache))
    end

ProbCircuitNode(::⋁, n::CircuitNode, cache::ProbCache) =
    get!(cache, n) do
        Prob⋁(n, ProbCircuit(n.children, cache), some_vector(Float64, num_children(n)))
    end

ProbCircuit(c::Circuit△, cache::ProbCache = ProbCache()) = map(n->ProbCircuitNode(n,cache), c)

#####################
# methods
#####################

import ..Logical: literal, children # make available for extension

@inline literal(n::ProbLiteral)::Lit  = literal(n.origin)
@inline children(n::ProbInnerNode) = n.children

num_parameters(n::Prob⋁) = num_children(n)
num_parameters(c::ProbCircuit△) = sum(n -> num_parameters(n), ⋁_nodes(c))

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

"""
Calculates log likelihood for a batch of fully observed samples.
(Also retures the generated FlowCircuit)
"""
function log_likelihood_per_instance(pc::ProbCircuit△, batch::PlainXData{Bool})
    opts = (flow_opts★..., el_type=Bool, compact⋁=false) #keep default options but insist on Bool flows
    fc = FlowCircuit(pc, num_examples(batch), Bool, FlowCache(), opts)
    (fc, log_likelihood_per_instance(fc, batch))
end

"""
Calculate log likelihood for a batch of fully observed samples.
(This is for when you already have a FlowCircuit)
"""
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
            log_likelihoods .+= prod_fast(downflow(n), pr_factors(c)) .* theta
        end
    end
end

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(Also returns the generated FlowCircuit)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(pc::ProbCircuit△, batch::PlainXData{Int8})
    opts = (flow_opts★..., el_type=Float64, compact⋁=false)
    fc = FlowCircuit(pc, num_examples(batch), Float64, FlowCache(), opts)
    (fc, marginal_log_likelihood_per_instance(fc, batch))
end

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(If you already have a FlowCircuit)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(fc::FlowCircuit△, batch::PlainXData{Int8})
    @assert (fc[end].origin isa ProbCircuitNode) "FlowCircuit must originate in a ProbCircuit"
    marginal_pass_up(fc, batch)
    pr(fc[end])
end

function check_parameter_integrity(circuit::ProbCircuit△)
    for node in circuit |> @filter(_ isa Prob⋁)
        @assert all(θ -> !isnan(θ), node.log_thetas) "There is a NaN in one of the log_thetas"
    end
    true
end

##################
# Sampling from a psdd
##################

"""
Sample from a PSDD without any evidence
"""
function sample(circuit::ProbCircuit△)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

# Uniformly sample based on the proability of the items
# and return the selected index
function sample(probs::AbstractVector)::Int32
    z = sum(probs)
    q = rand() * z
    cur = 0.0
    for i = 1:length(probs)
        cur += probs[i]
        if q <= cur
            return i
        end
    end
    return length(probs)
end

function simulate(node::ProbLiteral, inst::Dict{Var,Int64})
    if positive(node)
        inst[variable(node.origin)] = 1
    else
        inst[variable(node.origin)] = 0
    end
end
    
function simulate(node::Prob⋁, inst::Dict{Var,Int64})
    idx = sample(exp.(node.log_thetas))
    simulate(node.children[idx], inst)
end
function simulate(node::Prob⋀, inst::Dict{Var,Int64})
    for child in node.children
        simulate(child, inst)
    end    
end

"""
Sampling with Evidence from a psdd.
Internally would call marginal pass up on a newly generated flow circuit.
"""
function sample(circuit::ProbCircuit△, evidence::PlainXData{Int8})::AbstractVector{Bool}
    opts= (compact⋀=false, compact⋁=false)
    flow_circuit = FlowCircuit(circuit, 1, Float64, FlowCache(), opts)
    marginal_pass_up(flow_circuit, evidence)
    sample(flow_circuit)
end

"""
Sampling with Evidence from a psdd.
Assuming already marginal pass up has been done on the flow circuit.
"""
function sample(circuit::FlowCircuit△)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate2(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

function simulate2(node::Logical.FlowLiteral, inst::Dict{Var,Int64})
    if positive(node)
        inst[variable(node.origin.origin)] = 1
    else
        inst[variable(node.origin.origin)] = 0
    end
end

function simulate2(node::Logical.Flow⋁, inst::Dict{Var,Int64})
    prs = [ pr(ch)[1] for ch in children(node) ]
    idx = sample(exp.(node.origin.log_thetas .+ prs))
    simulate2(children(node)[idx], inst)
end

function simulate2(node::Logical.Flow⋀, inst::Dict{Var,Int64})
    for child in children(node)
        simulate2(child, inst)
    end    
end

