#######################
## Logistic Circuits
#######################


abstract type LogisticCircuitNode <: DecoratorCircuitNode end
abstract type LogisticLeafNode <: LogisticCircuitNode end
abstract type LogisticInnerNode <: LogisticCircuitNode end

struct LogisticLiteral <: LogisticLeafNode
    origin::CircuitNode
    thetas::Array{Float64, 1}
end

struct Logistic⋀ <: LogisticInnerNode
    origin::CircuitNode
    children::Vector{<:LogisticCircuitNode}
end

mutable struct Logistic⋁ <: LogisticInnerNode
    origin::CircuitNode
    children::Vector{<:LogisticCircuitNode}
    thetas::Array{Float64, 2}
end

mutable struct LogisticBias <: LogisticInnerNode
    biases::Array{Float64, 1}
end


const LogisticCircuit△ = AbstractVector{<:LogisticCircuitNode}

#####################
# traits
#####################

import ..Logical.NodeType # make available for extension

@inline NodeType(::Type{<:LogisticLiteral}) = LiteralLeaf()
@inline NodeType(::Type{<:Logistic⋀}) = ⋀()
@inline NodeType(::Type{<:Logistic⋁}) = ⋁()



#####################
# constructors and conversions
#####################

function Logistic⋁(origin, children, classes::Int)
    Logistic⋁(origin, children, Array{Float64, 2}(undef, (length(children), classes)))
end


const LogisticCache = Dict{CircuitNode, LogisticCircuitNode}

function LogisticCircuit(circuit::Circuit△, classes::Int, cache::LogisticCache = LogisticCache())

    sizehint!(cache, length(circuit)*4÷3)
    
    pc_node(::LiteralLeaf, n::CircuitNode) = LogisticLiteral(n, Array{Float64, 1}(undef, classes))
    pc_node(::ConstantLeaf, n::CircuitNode) = error("Cannot construct a logistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")

    pc_node(::⋀, n::CircuitNode) = begin
        children = map(c -> cache[c], n.children)
        Logistic⋀(n, children)
    end

    pc_node(::⋁, n::CircuitNode) = begin
        children = map(c -> cache[c], n.children)
        Logistic⋁(n, children, classes)
    end
        
    map(circuit) do node
        pcn = pc_node(NodeType(node), node)
        cache[node] = pcn
        pcn
    end
end


#####################
# methods
#####################

import ..Logical: literal, children # make available for extension

@inline literal(n::LogisticLiteral)::Lit  = literal(n.origin)
@inline children(n::LogisticInnerNode) = n.children
@inline classes(n::Logistic⋁) = if length(n.children) > 0 length(n.thetas[1]) else 0 end;

num_parameters(n::Logistic⋁) = num_children(n) * classes(n)
num_parameters(c::LogisticCircuit△) = sum(n -> num_parameters(n), ⋁_nodes(c))

num_parameters_perclass(n::Logistic⋁) = num_children(n)
num_parameters_perclass(c::LogisticCircuit△) = sum(n -> num_parameters_perclass(n), ⋁_nodes(c))

"Return the first origin that is a Logistic circuit node"
logistic_origin(n::DecoratorCircuitNode)::LogisticCircuitNode = logistic_origin(n.origin)
logistic_origin(n::LogisticCircuitNode)::LogisticCircuitNode = n

"Return the first origin that is a Logistic circuit"
logistic_origin(c::DecoratorCircuit△)::LogisticCircuit△ = logistic_origin(origin(c))
logistic_origin(c::LogisticCircuit△)::LogisticCircuit△ = c


# TODO Learning



# Class Conditional Probability
function class_conditional_likelihood_per_instance(fc::FlowCircuit△, 
                                                    classes::Int, 
                                                    batch::PlainXData{Bool})
    lc = origin(fc)
    @assert(lc isa LogisticCircuit△)
    pass_up_down(fc, batch)
    likelihoods = zeros(num_examples(batch), classes)
    for n in fc
        origin = logistic_origin(n)
        if origin isa Logistic⋁
            # For each class. origin.thetas is 2D so used eachcol
            for (idx, thetaC) in enumerate(eachcol(origin.thetas))
                foreach(n.children, thetaC) do c, theta
                    likelihoods[:, idx] .+= prod_fast(downflow(n), pr_factors(c)) .* theta
                end
            end
        elseif origin isa LogisticLiteral
            # For each class. origin.thetas is 1D so used eachrow
            for (idx, thetaC) in enumerate(eachrow(origin.thetas))
                likelihoods[:, idx] .+= n.pr .* thetaC
            end
        end
    end
    likelihoods
end

