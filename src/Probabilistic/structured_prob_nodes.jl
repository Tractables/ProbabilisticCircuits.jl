export ProbCircuit, StructProbCircuit,  StructProbLeafNode, StructProbInnerNode,
    StructProbLiteralNode, StructProb⋀Node, StructProb⋁Node, check_parameter_integrity

#####################
# Prob circuits that are structured,
# meaning that each conjunction is associated with a vtree node.
#####################

"Root of the plain structure probabilistic circuit node hierarchy"
abstract type StructProbCircuit <: StructLogicCircuit end

"Root of the probabilistic circuit node hierarchy"
const ProbCircuit = Union{StructProbCircuit, PlainProbCircuit}

"A plain structured probabilistic leaf node"
abstract type StructProbLeafNode <: StructProbCircuit end

"A plain structured probabilistic inner node"
abstract type StructProbInnerNode <: StructProbCircuit end

"A plain structured probabilistic literal leaf node, representing the positive or negative literal of its variable"
mutable struct StructProbLiteralNode <: StructProbLeafNode
    literal::Lit
    vtree::Vtree
    data
    counter::UInt32
    StructProbLiteralNode(l,v) = begin
        @assert lit2var(l) ∈ variables(v) 
        new(l, v, nothing, 0)
    end
end

"A plain structured probabilistic conjunction node"
mutable struct StructProb⋀Node <: StructProbInnerNode
    prime::StructProbCircuit
    sub::StructProbCircuit
    vtree::Vtree
    data
    counter::UInt32
    StructProb⋀Node(p,s,v) = begin
        @assert isinner(v) "Structured conjunctions must respect inner vtree node"
        @assert varsubset_left(vtree(p),v) "$p does not go left in $v"
        @assert varsubset_right(vtree(s),v) "$s does not go right in $v"
        new(p,s, v, nothing, 0)
    end
end

"A plain structured probabilistic disjunction node"
mutable struct StructProb⋁Node <: StructProbInnerNode
    children::Vector{<:StructProbCircuit}
    log_thetas::Vector{Float64}
    vtree::Vtree # could be leaf or inner
    data
    counter::UInt32
    StructProb⋁Node(c, v) = new(c, init_array(Float64, length(c)), v, nothing, 0)
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:StructProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:StructProb⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:StructProb⋁Node}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::StructProb⋁Node) = n.children
@inline children(n::StructProb⋀Node) = [n.prime,n.sub]

import ..Utils: num_parameters
@inline num_parameters(c::StructProbCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

#####################
# constructors and conversions
#####################

function StructProbCircuit(circuit::PlainStructLogicCircuit)::StructProbCircuit
    f_con(n) = error("Cannot construct a struct probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = StructProbLiteralNode(literal(n), vtree(n))
    f_a(n, cn) = begin
        @assert length(cn)==2
        StructProb⋀Node(cn[1], cn[2], vtree(n))
    end
    f_o(n, cn) = StructProb⋁Node(cn, vtree(n))
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, StructProbCircuit)
end 

function PlainStructLogicCircuit(circuit::StructProbCircuit)::PlainStructLogicCircuit
    f_con(n) = error("Cannot construct a struct probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = PlainStructLiteralNode(literal(n), vtree(n))
    f_a(n, cn) = begin
        @assert length(cn)==2
        PlainStruct⋀Node(cn[1], cn[2], vtree(n))
    end
    f_o(n, cn) = PlainStruct⋁Node(cn, vtree(n))
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, PlainStructLogicCircuit)
end

@inline ProbCircuit(circuit::PlainStructLogicCircuit) = StructProbCircuit(circuit)

@inline ProbCircuit(circuit::PlainLogicCircuit) = PlainProbCircuit(circuit)

# TODO: import LogicCircuits: conjoin, disjoin, compile # make available for extension 

function check_parameter_integrity(circuit::ProbCircuit)
    for node in or_nodes(circuit)
        @assert all(θ -> !isnan(θ), node.log_thetas) "There is a NaN in one of the log_thetas"
    end
    true
end
