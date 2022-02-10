export ProbCircuit,
    multiply, summate, 
    isinput, ismul, issum,
    inputnodes, mulnodes, sumnodes, 
    num_parameters, num_parameters_node, params,
    inputs, num_inputs,
    dist, randvar, randvars

const Var = UInt32

#####################
# Abstract probabilistic circuit nodes
#####################

"Root of the probabilistic circuit node hierarchy"
abstract type ProbCircuit <: DAG end

"Probabilistic circuit node types"
abstract type NodeType end
struct InputNode <: NodeType end
abstract type InnerNode <: NodeType end
struct SumNode <: InnerNode end
struct MulNode <: InnerNode end

"Get the probabilistic circuit node type"
NodeType(pc::ProbCircuit) = NodeType(typeof(pc))

DAGs.NodeType(::Type{T}) where {T<:ProbCircuit} =
    DAGs.NodeType(NodeType(T))
DAGs.NodeType(::InnerNode) = DAGs.Inner()
DAGs.NodeType(::InputNode) = DAGs.Leaf()

"Get the inputs of a PC node"
function inputs end

# DirectedAcyclicGraphs.jl has the convention that edges are directed away from the root
DAGs.children(pc::ProbCircuit) = inputs(pc)

"Get the distribution of a PC input node"
function dist end

"Get the parameters associated with a sum node"
params(n::ProbCircuit) = n.params

"Count the number of parameters in the node"
num_parameters_node(n) = 
    num_parameters_node(n, true) # default to independent = true

"Multiply nodes into a single circuit"
function multiply end

"Sum nodes into a single circuit"
function summate end

"Get the random variable associated with a PC input node"
function randvar end

#####################
# derived functions
#####################

"Is the node an input node?"
isinput(n) = (NodeType(n) isa InputNode)

"Is the node a multiplication?"
ismul(n) = (NodeType(n) isa MulNode)

"Is the node a summation?"
issum(n) = (NodeType(n) isa SumNode)

"Get all multiplication nodes in a given circuit"
inputnodes(pc) = filter(isinput, pc)

"Get all input nodes in a given circuit"
mulnodes(pc) = filter(ismul, pc)

"Get all summation nodes in a given circuit"
sumnodes(pc) = filter(issum, pc)

"Count the number of parameters in the circuit"
num_parameters(pc, independent = true) = 
    sum(n -> num_parameters_node(n, independent), sumnodes(pc))

"Number of inputs of a PC node"
num_inputs(pc) = num_inputs(pc, NodeType(pc))
num_inputs(_, ::InputNode) = 0
num_inputs(pc, ::InnerNode) = length(inputs(pc))

"""
    variables(pc::ProbCircuit)::BitSet

Get a bitset of variables mentioned in the circuit.
"""
function randvars(pc::ProbCircuit, cache = nothing)::BitSet
    f_leaf(n) = BitSet(variable(n))
    f_inner(n, call) = mapreduce(call, union, inputs(n))
    foldup(pc, f_leaf, f_inner, BitSet, cache)
end

"Number of variables in the data structure"
num_randvars(pc) = length(randvars(pc))

#####################
# constructor conveniences
#####################

multiply(xs::ProbCircuit...) = multiply(collect(xs))
summate(xs::ProbCircuit...) = summate(collect(xs))

Base.:*(x::ProbCircuit, y::ProbCircuit) = multiply(x,y)
Base.:*(xs::ProbCircuit...) = multiply(xs...)
Base.:+(x::ProbCircuit, y::ProbCircuit) = summate(x,y)
Base.:+(xs::ProbCircuit...) = summate(xs...)

# circuit construction with arithmetic operators
struct WeightProbCircuit 
    weight::Float32
    pc::ProbCircuit
end

Base.:*(w::Real, x::ProbCircuit) = WeightProbCircuit(w, x)
Base.:*(x::ProbCircuit, w::Real) = w * x

function Base.:+(x::WeightProbCircuit...)
    terms = collect(x)
    pc = summate(map(x -> x.pc, terms))
    params(pc) .= log.(map(x -> x.weight, terms))
    pc
end

#####################
# debugging tools
#####################

function check_parameter_integrity(circuit::ProbCircuit)
    for node in sumnodes(circuit)
        @assert all(θ -> !isnan(θ), params(node)) "There is a NaN in one of the PC parameters"
    end
    true
end