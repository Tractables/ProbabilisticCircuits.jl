export ProbCircuit,
    multiply, summate, ismul, issum,
    num_parameters, num_parameters_node,
    mul_nodes, sum_nodes, params

using LogicCircuits

#####################
# Abstract probabilistic circuit nodes
#####################

"Root of the probabilistic circuit node hierarchy"
abstract type ProbCircuit end

"Probabilistic circuit node types"
abstract type ProbNodeType end
struct InputProbNode <: ProbNodeType end
abstract type InnerProbNode <: ProbNodeType end
struct SumProbNode <: InputProbNode end
struct MulProbNode <: InputProbNode end

"Get the probabilistic circuit node type"
ProbNodeType(pc::ProbCircuit) = ProbNodeType(typeof(pc))

DirectedAcyclicGraphs.NodeType(::Type{T}) where {T<:ProbCircuit} =
    NodeType(ProbNodeType(pc))
DirectedAcyclicGraphs.NodeType(::InnerProbNode) = Inner()
DirectedAcyclicGraphs.NodeType(::InputProbNode) = Leaf()

"Get the parameters associated with a sum node"
params(n::ProbCircuit) = n.params

"Multiply nodes into a single circuit"
function multiply end

"Sum nodes into a single circuit"
function summate end

#####################
# derived functions
#####################

"Is the node a multiplication?"
ismul(n) = (ProbNodeType(n) isa MulProbNode)

"Is the node a summation?"
issum(n) = (ProbNodeType(n) isa SumProbNode)

"Get the list of multiplication nodes in a given circuit"
mul_nodes(pc) = filter(ismul, pc)

"Get the list of summation nodes in a given circuit"
sum_nodes(pc) = filter(issum, pc)

"Count the number of parameters in the circuit"
num_parameters(pc; independent=true) = 
    sum(n -> num_parameters_node(n; independent), sum_nodes(pc))

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
    for node in sum_nodes(circuit)
        @assert all(θ -> !isnan(θ), params(node)) "There is a NaN in one of the PC parameters"
    end
    true
end