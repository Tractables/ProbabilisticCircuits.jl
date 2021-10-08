export PlainProbCircuit, 
    PlainProbLeafNode, PlainProbInnerNode, 
    PlainProbLiteralNode, PlainMulNode, PlainSumNode

#####################
# Plain probabilistic circuit nodes
#####################

"Root of the plain probabilistic circuit node hierarchy"
abstract type PlainProbCircuit <: ProbCircuit end

"A probabilistic leaf node"
abstract type PlainProbLeafNode <: PlainProbCircuit end

"A probabilistic inner node"
abstract type PlainProbInnerNode <: PlainProbCircuit end

"A probabilistic literal node"
struct PlainProbLiteralNode <: PlainProbLeafNode
    literal::Lit
end

"A probabilistic conjunction node (multiplication node)"
mutable struct PlainMulNode <: PlainProbInnerNode
    children::Vector{PlainProbCircuit}
    PlainMulNode(children) = begin
        new(convert(Vector{PlainProbCircuit}, children))
    end
end

"A probabilistic disjunction node (summation node)"
mutable struct PlainSumNode <: PlainProbInnerNode
    children::Vector{PlainProbCircuit}
    log_probs::Vector{Float64}
end

PlainSumNode(c) =
    PlainSumNode(c, log.(ones(Float64, length(c)) / length(c)))

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension

@inline GateType(::Type{<:PlainProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:PlainMulNode}) = ⋀Gate()
@inline GateType(::Type{<:PlainSumNode}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::PlainProbInnerNode) = n.children

"Count the number of parameters in the node"
@inline num_parameters_node(n::PlainSumNode) = num_children(n)

#####################
# constructors and conversions
#####################

function multiply(arguments::Vector{<:PlainProbCircuit};
                 reuse=nothing)
    @assert length(arguments) > 0
    reuse isa PlainMulNode && children(reuse) == arguments && return reuse
    return PlainMulNode(arguments)
end

function summate(arguments::Vector{<:PlainProbCircuit};
                    reuse=nothing)
    @assert length(arguments) > 0
    reuse isa PlainSumNode && children(reuse) == arguments && return reuse
    node = PlainSumNode(arguments)
    if reuse isa PlainSumNode && num_children(reuse) == length(arguments)
        node.log_probs = reuse.log_probs
    end
    return node
end

# claim `PlainProbCircuit` as the default `ProbCircuit` implementation
compile(::Type{ProbCircuit}, args...) =
    compile(PlainProbCircuit, args...)

compile(::Type{<:PlainProbCircuit}, l::Lit) =
    PlainProbLiteralNode(l)

function compile(::Type{<:PlainProbCircuit}, circuit::LogicCircuit)
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = compile(PlainProbCircuit, literal(n))
    f_a(_, cns) = multiply(cns)
    f_o(_, cns) = summate(cns)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, PlainProbCircuit)
end

import LogicCircuits: fully_factorized_circuit #extend

fully_factorized_circuit(::Type{ProbCircuit}, n::Int) =
    fully_factorized_circuit(PlainProbCircuit, n)

function fully_factorized_circuit(::Type{<:PlainProbCircuit}, n::Int)
    ff_logic_circuit = fully_factorized_circuit(PlainLogicCircuit, n)
    compile(PlainProbCircuit, ff_logic_circuit)
end