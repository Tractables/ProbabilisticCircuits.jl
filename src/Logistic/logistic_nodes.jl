export 
    LogisticCircuit, 
    LogisticLeafNode, 
    LogisticInnerNode, 
    LogisticLiteral,
    Logistic⋀Node,
    Logistic⋁Node,
    classes,
    num_parameters_perclass
    # class_conditional_likelihood_per_instance,
    
#####################
# Infrastructure for logistic circuit nodes
#####################

"Root of the logistic circuit node hierarchy"
abstract type LogisticCircuit <: DecoratorCircuit end

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
    origin::LogicCircuit
    data
    counter::UInt32
    LogisticLiteral(n) = begin 
        @assert GateType(n) isa LiteralGate
        new(n, nothing, 0)
    end
end

"""
A logistic conjunction node (And node)
"""
mutable struct Logistic⋀Node <: LogisticInnerNode
    origin::LogicCircuit
    children::Vector{<:LogisticCircuit}
    data
    counter::UInt32
    Logistic⋀Node(n, children) = begin
        @assert GateType(n) isa ⋀Gate
        new(n, convert(Vector{LogisticCircuit}, children), nothing, 0)
    end
end

"""
A logistic disjunction node (Or node)
"""
mutable struct Logistic⋁Node <: LogisticInnerNode
    origin::LogicCircuit
    children::Vector{<:LogisticCircuit}
    thetas::Array{Float64, 2}
    data
    counter::UInt32
    Logistic⋁Node(n, children, class::Int) = begin
        @assert GateType(n) isa ⋁Gate
        new(n, convert(Vector{LogisticCircuit}, children), init_array(Float64, length(children), class), nothing, 0)
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

import ..Utils: origin, num_parameters
@inline origin(c::LogisticCircuit) = c.origin
@inline num_parameters(c::LogisticCircuit) = sum(n -> num_children(n) * classes(n), ⋁_nodes(c))

import LogicCircuits: children # make available for extension
@inline children(n::LogisticInnerNode) = n.children
@inline classes(n::Logistic⋁Node) = size(n.thetas)[2]

@inline num_parameters_perclass(c::LogisticCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

#####################
# constructors and conversions
#####################

function LogisticCircuit(circuit::LogicCircuit, classes::Int)
    f_con(n) = error("Cannot construct a logistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = LogisticLiteral(n)
    f_a(n, cn) = Logistic⋀Node(n, cn)
    f_o(n, cn) = Logistic⋁Node(n, cn, classes)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, LogisticCircuit)

end

#####################
# methods
#####################
# TODO Learning
# TODO queries
# Class Conditional Probability
# function class_conditional_likelihood_per_instance(fc::FlowΔ, 
#                                                     classes::Int, 
#                                                     batch::PlainXData{Bool})
#     lc = origin(origin(fc))
#     @assert(lc isa LogisticΔ)
#     pass_up_down(fc, batch)
#     likelihoods = zeros(num_examples(batch), classes)
#     for n in fc
#         orig = logistic_origin(n)
#         if orig isa Logistic⋀Node
#             # For each class. orig.thetas is 2D so used eachcol
#             for (idx, thetaC) in enumerate(eachcol(orig.thetas))
#                 foreach(n.children, thetaC) do c, theta
#                     likelihoods[:, idx] .+= prod_fast(downflow(n), pr_factors(origin(c))) .* theta
#                 end
#             end
#         end
#     end
#     likelihoods
# end

# """
# Calculate conditional log likelihood for a batch of samples with evidence P(c | x).
# (Also returns the generated FlowΔ)
# """
# function class_conditional_likelihood_per_instance(lc::LogisticΔ, 
#                                                     classes::Int, 
#                                                     batch::PlainXData{Bool})
#     opts = (max_factors = 2, compact⋀=false, compact⋁=false)
#     fc = FlowΔ(lc, num_examples(batch), Float64, opts)
#     (fc, class_conditional_likelihood_per_instance(fc, classes, batch))
# end

