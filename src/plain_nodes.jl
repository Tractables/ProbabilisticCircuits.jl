#####################
# Plain probabilistic circuit nodes
#####################

"Root of the plain probabilistic circuit node hierarchy"
abstract type PlainProbCircuit <: ProbCircuit end

"A probabilistic input node"
mutable struct PlainInputNode{D <: InputDist} <: PlainProbCircuit 
    randvars::BitSet
    dist::D
end

PlainInputNode(randvars, dist) = 
    PlainInputNode(BitSet(randvars), dist)
PlainInputNode(randvar::Integer, dist) = 
    PlainInputNode(BitSet([randvar]), dist)

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
    num_in = length(inputs)
    # initialize with uniform log-parameters
    params = zeros(Float32, num_in) .- log(num_in)
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

randvars(n::PlainInputNode) = n.randvars

num_parameters_node(n::PlainInputNode, independent) = 
    num_parameters(dist(n), independent)
num_parameters_node(n::PlainMulNode, _) = 0
num_parameters_node(n::PlainSumNode, independent) = 
    num_inputs(n) - (independent ? 1 : 0)

num_bpc_parameters(n::PlainInputNode) = 
    num_bpc_parameters(dist(n))

init_params(n::PlainInputNode, perturbation::Float32) = begin
    d = init_params(dist(n), perturbation)
    n.dist = d
end

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
