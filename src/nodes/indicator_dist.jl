export Indicator, Literal

#####################
# indicators or logical literals
#####################

"A input distribution node that places all probability on a single value"
struct Indicator{T} <: InputDist
    value::T
end

"A logical literal input distribution node"
const Literal = Indicator{Bool}

num_parameters(n::Indicator, independent) = 0

value(d::Indicator) = d.value

params(d::Indicator) = value(d)

bits(d::Indicator, _ = nothing) = d

unbits(d::Indicator, _ = nothing) = d

loglikelihood(d::Indicator, value, _ = nothing) =
    (d.value == value) ?  zero(Float32) : -Inf32

init_params(d::Indicator, _) = d

map_loglikelihood(d::Indicator, _= nothing) =
    zero(Float32)

map_state(d::Indicator, _ = nothing) = 
    d.value

init_heap_map_state!(d::Indicator, _ = nothing) =
    nothing # do nothing since don't need heap for indicators

init_heap_map_loglikelihood!(d::Indicator, _= nothing) =
    nothing # do nothing since don't need heap for indicators

sample_state(d::Indicator, threshold=nothing, heap=nothing) = 
    d.value

# no learning necessary for indicator distributions
flow(d::Indicator, value, node_flow, heap) = nothing
update_params(d::Indicator, heap, pseudocount, inertia) = nothing
clear_memory(d::Indicator, heap, rate) = nothing
clear_memory(d::InputDist, heap, rate) = nothing