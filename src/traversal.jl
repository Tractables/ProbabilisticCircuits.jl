import Base: foreach # extend

function foreach(pc::ProbCircuit, f_i::Function, 
                f_m::Function, f_s::Function, seen=nothing)
    f_inner(n) = issum(n) ? f_s(n) : f_m(n)
    foreach(node, f_i, f_inner, seen)
    nothing # returning nothing helps save some allocations and time
end

import DirectedAcyclicGraphs: foldup # extend

"""
    foldup(node::ProbCircuit, 
        f_i::Function, 
        f_m::Function, 
        f_s::Function)::T where {T}

Compute a function bottom-up on the circuit. 
`f_in` is called on input nodes, `f_m` is called on product nodes, and `f_s` is called on sum nodes.
Values of type `T` are passed up the circuit and given to `f_m` and `f_s` through a callback from the children.
"""
function foldup(node::ProbCircuit, f_i::Function, f_m::Function, f_s::Function, ::Type{T}, cache=nothing)::T where {T}
    f_inner(n, call) = issum(n) ? f_s(n, call)::T : f_m(n, call)::T
    foldup(node, f_i, f_inner, T, cache)::T
end

import DirectedAcyclicGraphs: foldup_aggregate # extend

"""
    foldup_aggregate(node::ProbCircuit, 
        f_i::Function, 
        f_m::Function, 
        f_s::Function, 
        ::Type{T})::T where T

Compute a function bottom-up on the circuit. 
`f_in` is called on input nodes, `f_m` is called on product nodes, and `f_s` is called on sum nodes.
Values of type `T` are passed up the circuit and given to `f_m` and `f_s` in an aggregate vector from the children.
"""
function foldup_aggregate(node::ProbCircuit, f_i::Function, f_m::Function, f_s::Function, ::Type{T}, cache=nothing) where T
    f_inner(n, cs) = issum(n) ? f_s(n, cs)::T : f_m(n, cs)::T
    foldup_aggregate(node, f_leaf::Function, f_inner::Function, T, cache)::T
end