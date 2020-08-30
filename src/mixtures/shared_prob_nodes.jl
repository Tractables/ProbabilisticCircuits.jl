export SharedProbCircuit, SharedProbLeafNode, SharedProbInnerNode, SharedProbLiteralNode,
SharedProb⋀Node, SharedProb⋁Node, num_components

#####################
# Probabilistic circuits which share the same structure
#####################

"""
Root of the probabilistic circuit node hierarchy
"""
abstract type SharedProbCircuit <: LogicCircuit end

"""
A probabilistic leaf node
"""
abstract type SharedProbLeafNode <: SharedProbCircuit end

"""
A probabilistic inner node
"""
abstract type SharedProbInnerNode <: SharedProbCircuit end

"""
A probabilistic literal node
"""
mutable struct SharedProbLiteralNode <: SharedProbLeafNode
    literal::Lit
    data
    counter::UInt32
    SharedProbLiteralNode(l) = new(l, nothing, 0)
end

"""
A probabilistic conjunction node (And node)
"""
mutable struct SharedProb⋀Node <: SharedProbInnerNode
    children::Vector{<:SharedProbCircuit}
    data
    counter::UInt32
    SharedProb⋀Node(children) = new(convert(Vector{SharedProbCircuit}, children), nothing, 0)
end

"""
A probabilistic disjunction node (Or node)
"""
mutable struct SharedProb⋁Node <: SharedProbInnerNode
    children::Vector{<:SharedProbCircuit}
    log_thetas::Matrix{Float64}
    data
    counter::UInt32
    SharedProb⋁Node(children, n_mixture) = begin
        new(convert(Vector{SharedProbCircuit}, children), init_array(Float64, length(children), n_mixture), nothing, 0)
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:SharedProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:SharedProb⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:SharedProb⋁Node}) = ⋁Gate()

#####################
# constructors and conversions
#####################

function SharedProbCircuit(circuit::LogicCircuit, num_mixture::Int64)::SharedProbCircuit
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = SharedProbLiteralNode(literal(n))
    f_a(n, cn) = SharedProb⋀Node(cn)
    f_o(n, cn) = SharedProb⋁Node(cn, num_mixture)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, SharedProbCircuit)
end


import LogicCircuits: children # make available for extension

@inline children(n::SharedProbInnerNode) = n.children
@inline num_components(n::SharedProbCircuit) = size(n.log_thetas)[2]
