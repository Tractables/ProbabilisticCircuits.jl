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
    if value isa AbstractFloat && d isa Literal
        (d.value) ? log(value) : log1p(-value)
    else
        (d.value == value) ?  zero(Float32) : -Inf32
    end

init_params(d::Indicator, _) = d

sample_state(d::Indicator, threshold=nothing, heap=nothing) = d.value
map_state(d::Indicator, _ = nothing) = d.value
map_loglikelihood(d::Indicator, _= nothing) = zero(Float32)

# do nothing since don't need heap for indicators
init_heap_map_state!(d::Indicator, _ = nothing) = nothing 
init_heap_map_loglikelihood!(d::Indicator, _= nothing) = nothing 

# no learning necessary for indicator distributions
flow(d::Indicator, value, node_flow, heap) = nothing
update_params(d::Indicator, heap, pseudocount, inertia) = nothing
clear_memory(d::Indicator, heap, rate) = nothing