using CUDA

export InputDist, Bernoulli, Categorical, loglikelihood

abstract type InputDist end

import Base: isapprox #extend

isapprox(x::InputDist, y::InputDist) = 
    typeof(x) == typeof(y) && params(x) ≈ params(y)


#######################################################
### Functions to implement for each new input type ####
#######################################################

"""
    num_parameters(n::InputDist, independent)

Returns number of parameters for the input dist.
- `independent`: whether to only count independent parameters
"""
num_parameters(n::InputDist, independent) = 
    error("Not implemented error: `num_parameters`, $(typeof(n))")

"""
    params(n::InputDist)

Returns paramters of the input dist.
"""
params(n::InputDist) = 
    error("Not implemented error: `params`, $(typeof(n))")

"""
    init_params(d::InputDist, perturbation::Float32)    

Returns a new distribution with same type with initialized parameters.
"""
init_params(d::InputDist, perturbation::Float32) =
    error("Not implemented error: `init_params`, $(typeof(n))")

"""
    bits(d::InputDist, heap) 

Appends the required memory for this input dist to the heap.

Used internally for moving from CPU to GPU.
"""
bits(d::InputDist, heap) =
    error("Not implemented error: `bits`, $(typeof(n))")

"""
    unbits(d::InputDist, heap)

Returns the InputDist struct from the heap. Note, each input dist type
needs to store where in the heap its paramters are to be able to do this.

Used internally for moving from GPU to CPU.
"""
unbits(d::InputDist, heap) =
    error("Not implemented error: `unbits`, $(typeof(n))")

"""
    loglikelihood(d::InputDist, value, heap)

Returns the `log( P(input_var == value) )` according to the InputDist.
"""
loglikelihood(d::InputDist, value, heap) = 
    error("Not implemented error: `loglikelihood`, $(typeof(n))")

"""
    sample_state(d::InputDist, threshold::Float32, heap)

Returns a sample from InputDist.
`Threshold` is a uniform random value in range (0, 1) given to this API by the sampleing algorithm
"""
sample_state(d::InputDist, threshold::Float32, heap) = 
    error("Not implemented error: `sample_state`, $(typeof(n))")

"""
    init_heap_map_state!(d::InputDist, heap)

Initializes the heap for the input dist. Called before running MAP queries.
"""
init_heap_map_state!(d::InputDist, heap)  =
    error("Not implemented error: `init_heap_map_state!`, $(typeof(n))")

"""
    init_heap_map_loglikelihood!(d::InputDist, heap)

Initializes the heap for the input dist. Called before running MAP queries.
"""
init_heap_map_loglikelihood!(d::InputDist, heap) = 
    error("Not implemented error: `init_heap_map_loglikelihood!`, $(typeof(n))")

"""
    map_state(d::InputDist, heap)

Returns the MAP state for the InputDist d
"""
map_state(d::InputDist, heap) =
    error("Not implemented error: `map_state`, $(typeof(n))")

"""
    map_loglikelihood(d::InputDist, heap)

Returns the MAP loglikelihoods the most likely state of the InputDist d
"""
map_loglikelihood(d::InputDist, heap) =
    error("Not implemented error: `map_loglikelihood`, $(typeof(n))")

"""
    flow(d::InputDist, value, node_flow, heap)

Updates the "flow" values in the `heap` for the input node.
"""
flow(d::InputDist, value, node_flow, heap) = 
    error("Not implemented error: `flow`, $(typeof(n))")

"""
    update_params(d::InputDist, heap, pseudocount, inertia)

Update the parameters of the InputDist using stored values 
on the `heap` and (`pseudocount`, `inertia`)
"""
update_params(d::InputDist, heap, pseudocount, inertia) =
    error("Not implemented error: `update_params`, $(typeof(n))")

"""
    clear_memory(d::InputDist, heap, rate)

Clears the accumulated flow values on the `heap` by multiplying it by `rate`.
`rate == 0.0` will be equivalent to initializing the value to 0.0.
"""
clear_memory(d::InputDist, heap, rate) =
    error("Not implemented error: `clear_memory`, $(typeof(n))")

#########################################

include("./indicator_dist.jl")
include("./categorical_dist.jl")

