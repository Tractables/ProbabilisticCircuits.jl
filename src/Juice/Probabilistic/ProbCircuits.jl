#####################
# Probabilistic circuits
#####################

abstract type ProbCircuitNode{O} <: DecoratorCircuitNode{O} end
abstract type ProbLeafNode{O} <: ProbCircuitNode{O} end
abstract type ProbInnerNode{O} <: ProbCircuitNode{O} end

struct ProbLiteral{O} <: ProbLeafNode{O}
    origin::O
end

struct Prob⋀{O}<: ProbInnerNode{O}
    origin::O
    children::Vector{<:ProbCircuitNode{O}}
end

mutable struct Prob⋁{O} <: ProbInnerNode{O}
    origin::O
    children::Vector{<:ProbCircuitNode{<:O}}
    log_thetas::Vector{Float64}
end

const ProbCircuit△{O} = AbstractVector{<:ProbCircuitNode{O}}

#####################
# traits
#####################

import ..Logical.NodeType # make available for extension

@inline NodeType(::Type{<:ProbLiteral}) = LiteralLeaf()
@inline NodeType(::Type{<:Prob⋀}) = ⋀()
@inline NodeType(::Type{<:Prob⋁}) = ⋁()

#####################
# constructors and conversions
#####################

function Prob⋁(origin::O, children::Vector{<:ProbCircuitNode{<:O}}) where {O<:CircuitNode}
    Prob⋁{O}(origin, children, some_vector(Float64, length(children)))
end


const ProbCache = Dict{CircuitNode, ProbCircuitNode}

function ProbCircuit(circuit::Circuit△, cache::ProbCache = ProbCache())

    sizehint!(cache, length(circuit)*4÷3)
    
    pc_node(::LiteralLeaf, n::CircuitNode) = ProbLiteral(n)
    pc_node(::ConstantLeaf, n::CircuitNode) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")

    pc_node(::⋀, n::CircuitNode) = begin
        children = map(c -> cache[c], n.children)
        Prob⋀(n, children)
    end

    pc_node(::⋁, n::CircuitNode) = begin
        children = map(c -> cache[c], n.children)
        Prob⋁(n, children)
    end
        
    map(circuit) do node
        pcn = pc_node(NodeType(node), node)
        cache[node] = pcn
        pcn
    end
end

#####################
# methods
#####################

import ..Logical: literal, children # make available for extension

@inline literal(n::ProbLiteral)::Lit  = literal(n.origin)
@inline children(n::ProbInnerNode) = n.children

num_parameters(n::Prob⋁) = num_children(n)
num_parameters(c::ProbCircuit△) = sum(n -> num_parameters(n), ⋁_nodes(c))

"Return the first origin that is a probabilistic circuit node"
prob_origin(n::DecoratorCircuitNode)::ProbCircuitNode = prob_origin(n.origin)
prob_origin(n::ProbCircuitNode)::ProbCircuitNode = n

"Return the first origin that is a probabilistic circuit"
prob_origin(c::DecoratorCircuit△)::ProbCircuit△ = prob_origin(origin(c))
prob_origin(c::ProbCircuit△)::ProbCircuit△ = c

function estimate_parameters(pc::ProbCircuit△, data::XBatches{Bool}; pseudocount::Float64)
    estimate_parameters(AggregateFlowCircuit(pc, aggr_weight_type(data)), data; pseudocount=pseudocount)
end

function estimate_parameters(afc::AggregateFlowCircuit△, data::XBatches{Bool}; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (afc[end].origin isa ProbCircuitNode) "AggregateFlowCircuit must originate in a ProbCircuit"
    collect_aggr_flows(afc, data)
    estimate_parameters_cached(afc; pseudocount=pseudocount)
    afc
end

function estimate_parameters(fc::FlowCircuit△, data::XBatches{Bool}; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (prob_origin(afc[end]) isa ProbCircuitNode) "FlowCircuit must originate in a ProbCircuit"
    collect_aggr_flows(fc, data)
    estimate_parameters_cached(origin(fc); pseudocount=pseudocount)
end

 # turns aggregate statistics into theta parameters
function estimate_parameters_cached(afc::AggregateFlowCircuit△; pseudocount::Float64)
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
    fc = FlowCircuit(pc, num_examples(batch), Bool)
    (fc, log_likelihood_per_instance(fc, batch))
end

"""
Calculate log likelihood per instance for batches of samples.
"""
function log_likelihood_per_instance(pc::ProbCircuit△, batches::XBatches{Bool})::Vector{Float64}
    mapreduce(b -> log_likelihood_per_instance(pc, b)[2], vcat, batches)
end

"""
Calculate log likelihood for a batch of fully observed samples.
(This is for when you already have a FlowCircuit)
"""
function log_likelihood_per_instance(fc::FlowCircuit△, batch::PlainXData{Bool})
    @assert (prob_origin(fc[end]) isa ProbCircuitNode) "FlowCircuit must originate in a ProbCircuit"
    pass_up_down(fc, batch)
    log_likelihoods = zeros(num_examples(batch))
    indices = some_vector(Bool, flow_length(fc))::BitVector
    for n in fc
         if n isa Flow⋁ && num_children(n) != 1 # other nodes have no effect on likelihood
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

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(Also returns the generated FlowCircuit)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(pc::ProbCircuit△, batch::PlainXData{Int8})
    opts = (flow_opts★..., el_type=Float64, compact⋁=false)
    fc = FlowCircuit(pc, num_examples(batch), Float64, opts)
    (fc, marginal_log_likelihood_per_instance(fc, batch))
end

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(If you already have a FlowCircuit)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(fc::FlowCircuit△, batch::PlainXData{Int8})
    @assert (prob_origin(fc[end]) isa ProbCircuitNode) "FlowCircuit must originate in a ProbCircuit"
    marginal_pass_up(fc, batch)
    pr(fc[end])
end

function check_parameter_integrity(circuit::ProbCircuit△)
    for node in filter(n -> NodeType(n) isa Prob⋁, circuit)
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
    flow_circuit = FlowCircuit(circuit, 1, Float64, opts)
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



##################
# Most Probable Explanation MPE of a psdd
##################

# function mpe(circuit::ProbCircuit△)::AbstractVector{Bool}
#     inst = Dict{Var,Int64}()
#     mpe_simulate(circuit[end], inst)
#     len = length(keys(inst))
#     ans = Vector{Bool}()
#     for i = 1:len
#         push!(ans, inst[i])
#     end
#     ans
# end
# function mpe_simulate(node::ProbLiteral, inst::Dict{Var,Int64})
#     if positive(node)
#         inst[variable(node.origin)] = 1
#     else
#         inst[variable(node.origin)] = 0
#     end
# end
# function mpe_simulate(node::Prob⋁, inst::Dict{Var,Int64})
#     idx = findmax(node.log_thetas)[2] # findmax -> (value, index)
#     mpe_simulate(node.children[idx], inst)
# end
# function mpe_simulate(node::Prob⋀, inst::Dict{Var,Int64})
#     for child in node.children
#         mpe_simulate(child, inst)
#     end    
# end