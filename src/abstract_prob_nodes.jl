export ProbCircuit,
    multiply, summate, ismul, issum,
    num_parameters,
    mul_nodes, sum_nodes

using LogicCircuits: LogicCircuit

#####################
# Abstract probabilistic circuit nodes
#####################

"Root of the probabilistic circuit node hierarchy"
abstract type ProbCircuit <: LogicCircuit end

#####################
# node functions that need to be implemented for each type of circuit
#####################

import LogicCircuits: children, compile # extend

"Multiply nodes into a single circuit"
function multiply end

"Sum nodes into a single circuit"
function summate end

#####################
# derived functions
#####################

"Is the node a multiplication?"
@inline ismul(n) = GateType(n) isa ⋀Gate
"Is the node a summation?"
@inline issum(n) = GateType(n) isa ⋁Gate

"Count the number of parameters in the circuit"
@inline num_parameters(c::ProbCircuit) = 
    sum(n -> num_parameters_node(n), sum_nodes(c))

#####################
# methods to easily construct circuits
#####################

@inline multiply(xs::ProbCircuit...) = multiply(collect(xs))
@inline summate(xs::ProbCircuit...) = summate(collect(xs))

import LogicCircuits: conjoin, disjoin # make available for extension

# alias conjoin/disjoin using mul/sum terminology
@inline conjoin(args::Vector{<:ProbCircuit}; reuse=nothing) = 
    multiply(args; reuse)
@inline disjoin(args::Vector{<:ProbCircuit}; reuse=nothing) = 
    summate(args; reuse)

@inline Base.:*(x::ProbCircuit, y::ProbCircuit) = multiply(x,y)
@inline Base.:+(x::ProbCircuit, y::ProbCircuit) = summate(x,y)

#####################
# circuit inspection
#####################

"Get the list of multiplication nodes in a given circuit"
mul_nodes(c::ProbCircuit) = ⋀_nodes(c)

"Get the list of summation nodes in a given circuit"
sum_nodes(c::ProbCircuit) = ⋁_nodes(c)
