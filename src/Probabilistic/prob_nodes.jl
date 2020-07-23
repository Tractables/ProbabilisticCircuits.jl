export ProbCircuit, ProbLeafNode, ProbInnerNode, ProbLiteralNode, Prob⋀Node, 
Prob⋁Node, num_parameters, check_parameter_integrity

#####################
# Infrastructure for probabilistic circuit nodes
#####################

"Root of the probabilistic circuit node hierarchy"
abstract type ProbCircuit <: DecoratorCircuit end

"""
A probabilistic leaf node
"""
abstract type ProbLeafNode <: ProbCircuit end

"""
A probabilistic inner node
"""
abstract type ProbInnerNode <: ProbCircuit end

"""
A probabilistic literal node
"""
mutable struct ProbLiteralNode <: ProbLeafNode
    origin::LogicCircuit
    data
    counter::UInt32
    ProbLiteralNode(n) = begin 
        @assert GateType(n) isa LiteralGate
        new(n, nothing, 0)
    end
end

"""
A probabilistic conjunction node (And node)
"""
mutable struct Prob⋀Node <: ProbInnerNode
    origin::LogicCircuit
    children::Vector{<:ProbCircuit}
    data
    counter::UInt32
    Prob⋀Node(n, children) = begin
        @assert GateType(n) isa ⋀Gate
        new(n, convert(Vector{ProbCircuit}, children), nothing, 0)
    end
end

"""
A probabilistic disjunction node (Or node)
"""
mutable struct Prob⋁Node <: ProbInnerNode
    origin::LogicCircuit
    children::Vector{<:ProbCircuit}
    log_thetas::Vector{Float64}
    data
    counter::UInt32
    Prob⋁Node(n, children) = begin
        @assert GateType(n) isa ⋁Gate
        new(n, convert(Vector{ProbCircuit}, children), init_array(Float64, length(children)), nothing, 0)
    end
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension

@inline GateType(::Type{<:ProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:Prob⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:Prob⋁Node}) = ⋁Gate()

#####################
# methods
#####################

import ..Utils.origin
@inline origin(c::ProbCircuit) = c.origin

import LogicCircuits: children # make available for extension
@inline children(n::ProbInnerNode) = n.children
@inline num_parameters(c::ProbCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

#####################
# constructors and conversions
#####################

function ProbCircuit(circuit::LogicCircuit)::ProbCircuit
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = ProbLiteralNode(n)
    f_a(n, cn) = Prob⋀Node(n, cn)
    f_o(n, cn) = Prob⋁Node(n, cn)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, ProbCircuit)
end

# TODO: import LogicCircuits: conjoin, disjoin, compile # make available for extension 


function check_parameter_integrity(circuit::ProbCircuit)
    for node in or_nodes(circuit)
        @assert all(θ -> !isnan(θ), node.log_thetas) "There is a NaN in one of the log_thetas"
    end
    true
end