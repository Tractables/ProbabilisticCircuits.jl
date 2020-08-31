export SharedProbCircuit, SharedPlainProbLeafNode, SharedPlainProbInnerNode, SharedPlainProbLiteralNode,
SharedPlainMulNode, SharedPlainSumNode, num_components

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
abstract type SharedPlainProbLeafNode <: SharedProbCircuit end

"""
A probabilistic inner node
"""
abstract type SharedPlainProbInnerNode <: SharedProbCircuit end

"""
A probabilistic literal node
"""
mutable struct SharedPlainProbLiteralNode <: SharedPlainProbLeafNode
    literal::Lit
    data
    counter::UInt32
    SharedPlainProbLiteralNode(l) = new(l, nothing, 0)
end

"""
A probabilistic conjunction node (And node)
"""
mutable struct SharedPlainMulNode <: SharedPlainProbInnerNode
    children::Vector{<:SharedProbCircuit}
    data
    counter::UInt32
    SharedPlainMulNode(children) = new(convert(Vector{SharedProbCircuit}, children), nothing, 0)
end

"""
A probabilistic disjunction node (Or node)
"""
mutable struct SharedPlainSumNode <: SharedPlainProbInnerNode
    children::Vector{<:SharedProbCircuit}
    log_probs::Matrix{Float64}
    data
    counter::UInt32
    SharedPlainSumNode(children, n_mixture) = begin
        new(convert(Vector{SharedProbCircuit}, children), init_array(Float64, length(children), n_mixture), nothing, 0)
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:SharedPlainProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:SharedPlainMulNode}) = ⋀Gate()
@inline GateType(::Type{<:SharedPlainSumNode}) = ⋁Gate()

#####################
# constructors and conversions
#####################

function SharedProbCircuit(circuit::LogicCircuit, num_mixture::Int64)::SharedProbCircuit
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = SharedPlainProbLiteralNode(literal(n))
    f_a(n, cn) = SharedPlainMulNode(cn)
    f_o(n, cn) = SharedPlainSumNode(cn, num_mixture)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, SharedProbCircuit)
end


import LogicCircuits: children # make available for extension

@inline children(n::SharedPlainProbInnerNode) = n.children
@inline num_components(n::SharedProbCircuit) = size(n.log_probs)[2]
