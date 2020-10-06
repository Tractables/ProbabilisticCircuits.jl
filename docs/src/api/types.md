# [Type Trees](@id api-types)

The following code snippet provides an easy way to print the type tree of probabilistic circuits.

```@example types
using InteractiveUtils;
using ProbabilisticCircuits;
using AbstractTrees;
AbstractTrees.children(x::Type) = subtypes(x);
```

For example, we can see [`ProbabilisticCircuits.ProbCircuit`](@ref)'s type tree.

```@example types
AbstractTrees.print_tree(ProbCircuit)
```

Alternatively, here's [`ProbabilisticCircuits.LogisticCircuit`](@ref)'s type tree.

```@example types
AbstractTrees.print_tree(LogisticCircuit)
```