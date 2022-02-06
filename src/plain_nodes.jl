#####################
# Plain probabilistic circuit nodes
#####################

"Root of the plain probabilistic circuit node hierarchy"
abstract type PlainProbCircuit <: ProbCircuit end

"A probabilistic input node"
struct PlainInputNode{D <: InputDist} <: PlainProbCircuit 
    randvar::Var
    dist::D
end

PlainInputNode(randvar, dist) = 
    PlainInputNode(convert(UInt32,randvar), dist)

"A probabilistic inner node"
abstract type PlainInnerNode <: PlainProbCircuit end

"A probabilistic multiplication node"
mutable struct PlainMulNode <: PlainInnerNode
    inputs::Vector{PlainProbCircuit}
end

"A probabilistic summation node"
mutable struct PlainSumNode <: PlainInnerNode
    inputs::Vector{PlainProbCircuit}
    params::Vector{Float32}
end

function PlainSumNode(inputs)
    n = length(inputs)
    # initialize with uniform log-parameters
    params = zeros(Float32, n) .- log(n)
    PlainSumNode(inputs, params)
end

#####################
# traits
#####################

NodeType(::Type{<:PlainInputNode}) = InputNode()
NodeType(::Type{<:PlainMulNode}) = MulNode()
NodeType(::Type{<:PlainSumNode}) = SumNode()

#####################
# methods
#####################

inputs(n::PlainInnerNode) = n.inputs
dist(n::PlainInputNode) = n.dist
randvar(n::PlainInputNode) = n.randvar

num_parameters_node(n::PlainInputNode, independent) = 
    num_parameters(dist(n), independent)
num_parameters_node(n::PlainMulNode, _) = 0
num_parameters_node(n::PlainSumNode, independent) = 
    length(params(n)) - (independent ? 1 : 0)

#####################
# constructors and conversions
#####################

function multiply(args::Vector{<:PlainProbCircuit}; reuse=nothing)
    @assert length(args) > 0
    if reuse isa PlainMulNode && inputs(reuse) == args 
        reuse
    else
        PlainMulNode(args)
    end
end

function summate(args::Vector{<:PlainProbCircuit}; reuse=nothing)
    @assert length(args) > 0
    if reuse isa PlainSumNode && inputs(reuse) == args
        reuse
    else
        PlainSumNode(args)
    end
end

function PlainProbCircuit(pc::ProbCircuit)
    f_i(n) = PlainInputNode(randvar(n), dist(n))
    f_m(_, ins) = multiply(ins)
    f_s(_, ins) = summate(ins)
    foldup_aggregate(pc, f_i, f_m, f_s, PlainProbCircuit)
end
