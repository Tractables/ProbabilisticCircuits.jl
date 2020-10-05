export ProbCircuit,
    multiply, summate, ismul, issum,
    num_parameters, num_parameters_node,
    mul_nodes, sum_nodes, params

using LogicCircuits

#####################
# Abstract probabilistic circuit nodes
#####################

"Root of the probabilistic circuit node hierarchy"
abstract type ProbCircuit <: LogicCircuit end

#####################
# node functions that need to be implemented for each type of probabilistic circuit
#####################

"Get the parameters associated with a sum node"
params(n::ProbCircuit) = n.log_probs

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

@inline Base.:*(x::ProbCircuit, y::ProbCircuit) = conjoin(x,y)
@inline Base.:*(xs::ProbCircuit...) = conjoin(xs...)
@inline Base.:+(x::ProbCircuit, y::ProbCircuit) = disjoin(x,y)
@inline Base.:+(xs::ProbCircuit...) = disjoin(xs...)

compile(::Type{<:ProbCircuit}, ::Bool) =
    error("Probabilistic circuits do not have constant leafs.")

struct WeightProbCircuit
    tmp_weight :: Float64
    circuit :: ProbCircuit
end

@inline Base.:*(w::Real, x::ProbCircuit) = WeightProbCircuit(w, x)
@inline Base.:*(x::ProbCircuit, w::Real) = w * x
@inline Base.:+(x::WeightProbCircuit...) = begin
    ch = collect(x)
    c = map(x -> x.circuit, ch)
    w = map(x -> x.tmp_weight, ch)
    pc = summate(c)
    pc.log_probs .= log.(w)
    pc
end

#####################
# circuit inspection
#####################

"Get the list of multiplication nodes in a given circuit"
mul_nodes(c::ProbCircuit) = ⋀_nodes(c)

"Get the list of summation nodes in a given circuit"
sum_nodes(c::ProbCircuit) = ⋁_nodes(c)

function check_parameter_integrity(circuit::ProbCircuit)
    for node in sum_nodes(circuit)
        @assert all(θ -> !isnan(θ), node.log_probs) "There is a NaN in one of the log_probs"
    end
    true
end