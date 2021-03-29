export 
    LogisticCircuit,
    LogisticLeafNode, LogisticInnerNode, 
    LogisticLiteralNode, Logistic⋀Node, Logistic⋁Node,
    num_classes, num_parameters_per_class
    
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
struct LogisticLiteralNode <: LogisticLeafNode
    literal::Lit
end

"""
A logistic conjunction node (And node)
"""
mutable struct Logistic⋀Node <: LogisticInnerNode
    children::Vector{<:LogisticCircuit}
    Logistic⋀Node(children) = begin
        new(convert(Vector{LogisticCircuit}, children))
    end
end

"""
A logistic disjunction node (Or node)
"""
mutable struct Logistic⋁Node <: LogisticInnerNode
    children::Vector{<:LogisticCircuit}
    thetas::Matrix{Float32}
    Logistic⋁Node(children, class::Int) = begin
        new(convert(Vector{LogisticCircuit}, children), init_array(Float32, length(children), class))
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:LogisticLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:Logistic⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:Logistic⋁Node}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::LogisticInnerNode) = n.children
@inline num_classes(n::Logistic⋁Node) = size(n.thetas)[2]

@inline num_parameters(c::LogisticCircuit) = sum(n -> num_children(n) * num_classes(n), ⋁_nodes(c))
@inline num_parameters_per_class(c::LogisticCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

"Get the parameters associated with a or node"
params(n::Logistic⋁Node) = n.thetas

#####################
# constructors and conversions
#####################

function LogisticCircuit(circuit::LogicCircuit, classes::Int)
    f_con(n) = error("Cannot construct a logistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = compile(LogisticCircuit, literal(n))
    f_a(n, cn) = Logistic⋀Node(cn)
    f_o(n, cn) = Logistic⋁Node(cn, classes)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, LogisticCircuit)
end

compile(::Type{<:LogisticCircuit}, l::Lit) = 
    LogisticLiteralNode(l)

function compile(::Type{<:LogisticCircuit}, classes, circuit::LogicCircuit)
    LogisticCircuit(circuit, classes)
end

import LogicCircuits: fully_factorized_circuit #extend

function fully_factorized_circuit(::Type{<:LogisticCircuit}, n::Int; classes::Int)
    ff_logic_circuit = fully_factorized_circuit(PlainLogicCircuit, n)
    compile(LogisticCircuit, classes, ff_logic_circuit)
end

function check_parameter_integrity(circuit::LogisticCircuit)
    for node in or_nodes(circuit)
        @assert all(θ -> !isnan(θ), node.thetas) "There is a NaN in one of the log_probs"
    end
    true
end