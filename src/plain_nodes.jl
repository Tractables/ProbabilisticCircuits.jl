#####################
# Plain probabilistic circuit nodes
#####################

"Root of the plain probabilistic circuit node hierarchy"
abstract type PlainProbCircuit <: ProbCircuit end

"A probabilistic input node"
mutable struct PlainInputNode{D <: InputDist} <: PlainProbCircuit 
    global_id::UInt32
    randvars::BitSet
    dist::D
end

PlainInputNode(randvars::BitSet, dist) = 
    PlainInputNode(zero(UInt32), randvars, dist)
PlainInputNode(randvars::Vector, dist) = 
    PlainInputNode(zero(UInt32), BitSet(randvars), dist)
PlainInputNode(randvar::Integer, dist) = 
    PlainInputNode(zero(UInt32), BitSet([randvar]), dist)

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
global_id(n::PlainInputNode) = n.global_id
dist(n::PlainInputNode) = n.dist
randvar(n::PlainInputNode) = begin
    @assert length(n.randvars) == 1 "Calling `randvar` on an input node with more than 1 variable"
    collect(n.randvars)[1]
end
randvars(n::PlainInputNode) = n.randvars
num_randvars(n::PlainInputNode) = length(n.randvars)

num_parameters_node(n::PlainInputNode, independent) = 
    num_parameters(dist(n), independent)
num_parameters_node(n::PlainMulNode, _) = 0
num_parameters_node(n::PlainSumNode, independent) = 
    length(params(n)) - (independent ? 1 : 0)

num_bpc_parameters(n::PlainInputNode) = 
    num_bpc_parameters(dist(n))

function assign_input_node_ids!(pc::PlainProbCircuit)
    global_id::UInt32 = zero(UInt32)
    foreach(pc) do n
        if n isa PlainInputNode
            global_id += one(UInt32)
            n.global_id = global_id
        end
    end
    nothing
end

function max_nvars_per_input(pc::PlainProbCircuit)
    max_nvars::UInt32 = zero(UInt32)
    foreach(pc) do n
        if n isa PlainInputNode
            if num_randvars(n) > max_nvars
                max_nvars = num_randvars(n)
            end
        end
    end
    max_nvars
end
function max_nparams_per_input(pc::PlainProbCircuit)
    max_nparams::UInt32 = zero(UInt32)
    foreach(pc) do n
        if n isa PlainInputNode
            if num_parameters_node(n, true) > max_nparams
                max_nparams = num_parameters_node(n, true)
            end
        end
    end
    max_nparams
end
function max_nedgeaggrs_per_input(pc::PlainProbCircuit)
    max_nparams::UInt32 = zero(UInt32)
    foreach(pc) do n
        if n isa PlainInputNode
            if num_bpc_parameters(n) > max_nparams
                max_nparams = num_bpc_parameters(n)
            end
        end
    end
    max_nparams
end

function num_input_nodes(pc::PlainProbCircuit)
    ninput_nodes::UInt32 = zero(UInt32)
    foreach(pc) do n
        if n isa PlainInputNode
            ninput_nodes += one(UInt32)
        end
    end
    ninput_nodes
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
