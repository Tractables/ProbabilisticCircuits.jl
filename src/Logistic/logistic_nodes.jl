export 
    LogisticCircuit,
    LogisticLeafNode, LogisticInnerNode, 
    LogisticLiteral, Logistic⋀Node, Logistic⋁Node,
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
    thetas::Matrix{Float64}
    data
    counter::UInt32
    Logistic⋁Node(children, class::Int) = begin
        new(convert(Vector{LogisticCircuit}, children), init_array(Float32, length(children), class), nothing, 0)
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
@inline num_classes(n::Logistic⋁Node) = size(n.thetas)[2]

import ..Utils: num_parameters
@inline num_parameters(c::LogisticCircuit) = sum(n -> num_children(n) * classes(n), ⋁_nodes(c))
@inline num_parameters_per_class(c::LogisticCircuit) = sum(n -> num_children(n), ⋁_nodes(c))



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



"""
Construct a `BitCircuit` while storing edge parameters in a separate array
"""
function ParamBitCircuit(lc::LogisticCircuit, nc, data)
    thetas::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n)
            # @assert first_element == last_element
            push!(thetas, zeros(Float64, nc))
        else
            # @assert last_element-first_element+1 == length(n.log_probs) "$last_element-$first_element+1 != $(length(n.log_probs))"
            for theta in eachrow(n.thetas)
                push!(thetas, theta)
            end
        end
    end
    bc = BitCircuit(lc, data; on_decision)
    ParamBitCircuit(bc, permutedims(hcat(params...), (2, 1)))
end