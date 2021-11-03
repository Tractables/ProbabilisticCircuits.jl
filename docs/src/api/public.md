# [Public APIs](@id api-public)

This page lists documentation for the most commonly used public APIs of `ProbabilisticCircuits.jl`. Visit the internals section for a auto generated documentation for more public API and internal APIs.

```@contents
Pages = ["public.md"]
```

## Circuit IO

```@docs
read
write
```

## Learning Circuits

```@docs
learn_circuit
learn_strudel
estimate_parameters!
estimate_parameters_em!
estimate_parameters_em_multi_epochs!
learn_chow_liu_tree_circuit
```

## Circuit Queries

```@docs
marginal
max_a_posteriori
sample
```

## Compilation

```@docs
compile_sdd_from_clt
```
