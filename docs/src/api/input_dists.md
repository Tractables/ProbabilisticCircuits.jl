# [Input Distributions](@id new-input-dist)

Currently we support `Indicator{T}`, `Categorical`, `Bernoulli` (special case of Categorical) distributions in the InputNodes.

#### Support new InputDist
To support new type of Input Distributions you need to implement
the following functions:

```julia
num_parameters
params
init_params
loglikelihood
```

#### Support movement between CPU/GPU for InputDist

To support moving between CPU/GPU you need to implement the following:

```julia
bits
unbits
```

#### Learning support for InputDist

To support learning you need to implement the following:

```julia
flow
update_params
clear_memory
```

#### Query support for InputDist

To support certain queries such as sampling and MAP you need to implement the following:

```julia
sample_state
init_heap_map_state!
init_heap_map_loglikelihood!
map_state
map_loglikelihood
```
