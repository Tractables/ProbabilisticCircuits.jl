# [Public APIs](@id api-public)

This page lists documentation for the most commonly used public APIs of `ProbabilisticCircuits.jl`. Visit the internals section for a auto generated documentation for more public API and internal APIs.

```@contents
Pages = ["public.md"]
```

## Loading Circuits

```@docs
load_prob_circuit
load_struct_prob_circuit
load_logistic_circuit
```

## Saving Interface

```@docs
save_circuit
save_as_psdd
save_as_logistic
save_as_dot
```

## Learning Circuits

```@docs
learn_parameters
learn_chow_liu_tree_circuit
learn_circuit
learn_strudel
learn_circuit_mixture
```

## Circuit Queries

```@docs
marginal
max_a_posteriori
```

## Compilation

```@docs
compile_sdd_from_clt
```
