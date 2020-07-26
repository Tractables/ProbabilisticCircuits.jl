export 
    LogisticCircuit, 
    LogisticLeafNode, 
    LogisticInnerNode, 
    LogisticLiteral,
    Logistic⋀Node,
    Logistic⋁Node,
    classes,
    num_parameters_perclass
    
#####################
# Infrastructure for logistic circuit nodes
#####################

"Root of the logistic circuit node hierarchy"
abstract type LogisticCircuit <: LogicCircuit end

"""
A logistic leaf node
"""
abstract type LogisticLeafNode <: LogisticCircuit end

"""
A logistic inner node
"""
abstract type LogisticInnerNode <: LogisticCircuit end

"""
A logistic literal node
"""
mutable struct LogisticLiteral <: LogisticLeafNode
    literal::Lit
    data
    counter::UInt32
    LogisticLiteral(l) = begin 
        new(l, nothing, 0)
    end
end

"""
A logistic conjunction node (And node)
"""
mutable struct Logistic⋀Node <: LogisticInnerNode
    children::Vector{<:LogisticCircuit}
    data
    counter::UInt32
    Logistic⋀Node(children) = begin
        new(convert(Vector{LogisticCircuit}, children), nothing, 0)
    end
end

"""
A logistic disjunction node (Or node)
"""
mutable struct Logistic⋁Node <: LogisticInnerNode
    children::Vector{<:LogisticCircuit}
    thetas::Array{Float64, 2}
    data
    counter::UInt32
    Logistic⋁Node(children, class::Int) = begin
        new(convert(Vector{LogisticCircuit}, children), init_array(Float64, length(children), class), nothing, 0)
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:LogisticLiteral}) = LiteralGate()
@inline GateType(::Type{<:Logistic⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:Logistic⋁Node}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::LogisticInnerNode) = n.children
@inline classes(n::Logistic⋁Node) = size(n.thetas)[2]

import ..Utils: num_parameters
@inline num_parameters(c::LogisticCircuit) = sum(n -> num_children(n) * classes(n), ⋁_nodes(c))
@inline num_parameters_perclass(c::LogisticCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

#####################
# constructors and conversions
#####################

function LogisticCircuit(circuit::LogicCircuit, classes::Int)
    f_con(n) = error("Cannot construct a logistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = LogisticLiteral(literal(n))
    f_a(n, cn) = Logistic⋀Node(cn)
    f_o(n, cn) = Logistic⋁Node(cn, classes)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, LogisticCircuit)
end
