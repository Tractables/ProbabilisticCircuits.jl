export ProbCircuit,
    multiply, summate, ismul, issum,
    num_parameters, num_parameters_node,
    mulnodes, sumnodes, params,
    inputs, num_inputs


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

# inputs correspond to reverse directed edges 
# TODO: swap the meaning of children and parents in DirectedAcyclicGraphs.jl
DAGs.children(pc::ProbCircuit) = inputs(pc)

"Get the parameters associated with a sum node"
function params end
params(n::ProbCircuit) = n.params

"Count the number of parameters in the node"
num_parameters_node(n) = 
    num_parameters_node(n, true) # default to independent = true

"Multiply nodes into a single circuit"
function multiply end

"Sum nodes into a single circuit"
function summate end

#####################
# derived functions
#####################

"Is the node a multiplication?"
ismul(n) = (NodeType(n) isa MulNode)

"Is the node a summation?"
issum(n) = (NodeType(n) isa SumNode)

"Get the list of multiplication nodes in a given circuit"
mulnodes(pc) = filter(ismul, pc)

"Get the list of summation nodes in a given circuit"
sumnodes(pc) = filter(issum, pc)

"Count the number of parameters in the circuit"
num_parameters(pc, independent = true) = 
    sum(n -> num_parameters_node(n, independent), sumnodes(pc))

"Number of inputs of a PC node"
num_inputs(pc) = num_inputs(pc, NodeType(pc))
num_inputs(_, ::InputNode) = 0
num_inputs(pc, ::InnerNode) = length(inputs(pc))

#####################
# constructor conveniences
#####################

multiply(xs::ProbCircuit...) = multiply(collect(xs))
summate(xs::ProbCircuit...) = summate(collect(xs))

Base.:*(x::ProbCircuit, y::ProbCircuit) = multiply(x,y)
Base.:*(xs::ProbCircuit...) = multiply(xs...)
Base.:+(x::ProbCircuit, y::ProbCircuit) = summate(x,y)
Base.:+(xs::ProbCircuit...) = summate(xs...)

#####################
# debugging tools
#####################

function check_parameter_integrity(circuit::ProbCircuit)
    for node in sumnodes(circuit)
        @assert all(θ -> !isnan(θ), params(node)) "There is a NaN in one of the PC parameters"
    end
    true
end