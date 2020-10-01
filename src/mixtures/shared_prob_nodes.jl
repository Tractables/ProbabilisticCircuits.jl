export SharedProbCircuit, SharedProbLeafNode, SharedProbInnerNode, SharedProbLiteralNode,
SharedMulNode, SharedSumNode, num_components

#####################
# Probabilistic circuits which share the same structure
#####################

"""
Root of the shared probabilistic circuit node hierarchy
"""
abstract type SharedProbCircuit <: ProbCircuit end

"""
A shared probabilistic leaf node
"""
abstract type SharedProbLeafNode <: SharedProbCircuit end

"""
A shared probabilistic inner node
"""
abstract type SharedProbInnerNode <: SharedProbCircuit end

"""
A shared probabilistic literal node
"""
mutable struct SharedProbLiteralNode <: SharedProbLeafNode
    literal::Lit
    data
    counter::UInt32
    SharedProbLiteralNode(l) = new(l, nothing, 0)
end

"""
A shared probabilistic multiplcation node
"""
mutable struct SharedMulNode <: SharedProbInnerNode
    children::Vector{<:SharedProbCircuit}
    data
    counter::UInt32
    SharedMulNode(children) = new(children, nothing, 0)
end

"""
A shared probabilistic summation node
"""
mutable struct SharedSumNode <: SharedProbInnerNode
    children::Vector{<:SharedProbCircuit}
    log_probs::Matrix{Float64}
    data
    counter::UInt32
    SharedSumNode(children, n_mixture) = begin
        new(children, init_array(Float64, length(children), n_mixture), nothing, 0)
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:SharedProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:SharedMulNode}) = ⋀Gate()
@inline GateType(::Type{<:SharedSumNode}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::SharedProbInnerNode) = n.children

@inline num_parameters_node(n::SharedSumNode) = length(n.log_probs)

"How many components are mixed together in this shared circuit?"
@inline num_components(n::SharedSumNode) = size(n.log_probs,2)

#####################
# constructors and conversions
#####################

function multiply(arguments::Vector{<:SharedProbCircuit};
    reuse=nothing)
    @assert length(arguments) > 0
    reuse isa SharedMulNode && children(reuse) == arguments && return reuse
    return SharedMulNode(arguments)
end

function summate(arguments::Vector{<:SharedProbCircuit}, num_components=0;
       reuse=nothing)
    @assert length(arguments) > 0
    reuse isa SharedSumNode && children(reuse) == arguments && return reuse
    return SharedSumNode(arguments, num_components) # unknwown number of components; resize later
end

compile(::Type{<:SharedProbCircuit}, l::Lit) =
    SharedProbLiteralNode(l)

function compile(::Type{<:SharedProbCircuit}, circuit::LogicCircuit, num_components::Int)
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = compile(SharedProbCircuit, literal(n))
    f_a(_, cns) = multiply(cns)
    f_o(_, cns) = summate(cns, num_components)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, SharedProbCircuit)
end

import LogicCircuits: fully_factorized_circuit #extend

function fully_factorized_circuit(::Type{<:SharedProbCircuit}, n::Int)
    ff_logic_circuit = fully_factorized_circuit(PlainLogicCircuit, n)
    compile(SharedProbCircuit, ff_logic_circuit)
end
