# [Quick Demo](@id man-demo)


In this section, we provide quick code snippets to get started with ProbabilisticCircuits and provide basic understanding of them. PCs are represented as a computational graphs that define a joint probability distribution as recursive mixtures (sum units) and factorizations (product units) of simpler distributions (input units).

Generally, we learn structure and paramters of circuit from data. Alternatively, we can also specify circuits in code. For example, the following snippet defines a circuit depending on 3 random variables. The `literals` function returns the input units of the circuit, in this case we get 6 different units (3 for positive literals, and 3 for negative literlas).  You can use `*` and `+` operators to build a circuits.

```@example demo
using LogicCircuits;
using ProbabilisticCircuits;

X1, X2, X3 = literals(ProbCircuit, 3)
pc = 0.3 * (X1[1] *
             (0.2 * X2[1] + 0.8 * X3[2])) +
     0.7 * (X1[2] *
             (0.4 * X2[2] + 0.6 * X3[1]));

nothing # hide
```

We can also plot circuits using `plot(pc)` to see the computation graph (structure and paramters). The output of `plot(pc)` has a type of `TikzPictures.TikzPicture`. Generally, notebooks automatically renders it and you see the figure in the notebook. 

```@example demo
using TikzPictures  # hide
TikzPictures.standaloneWorkaround(true)  # hide
plot(pc);
```

However, if you are not using a notebook or want to save to file you can use the following commands to save the plot in various formats.

```julia
using TikzPictures;
z = plot(pc);
save(PDF("plot"), z);
save(SVG("plot"), z);
save(TEX("plot"), z);
save(TIKZ("plot"), z);
```

You can ask basic questions about PCs, such as (1) how many variables they depends on, (2) how many nodes, (3) how many edges, (4) or how many parameters they have.

```@example demo
num_variables(pc)
```

```@example demo
num_nodes(pc)
```

```@example demo
num_edges(pc)
```

```@example demo
num_parameters(pc)
```

In the case that we have literals as input units, PCs can also be thought of as adding paramters to a LogicCircuit to define a distribution (See `LogicCircuit.jl` docs for more details). To get the corresponding logical formula, we can:

```@example demo
tree_formula_string(pc)
```

To enable tractable queries and opertations, PCs need to have certain structural properties. For example, we can check for smoothness and determinism as follows:

```@example demo
c1 = 0.4 * X1[1] + 0.6 * X1[2];
"Is $(tree_formula_string(c1)) smooth? $(issmooth(c1))"

```

```@example demo
c2 = 0.4 * X1[1] + 0.6 * X2[2];
"Is $(tree_formula_string(c2)) smooth? $(issmooth(c2))"
```

```@example demo
c1 = X1[1] * X2[1] + X1[1] * X2[2];
"Is $(tree_formula_string(c1)) deterministic? $(isdeterministic(c1))" 
```

```@example demo
c2 = X1[1] * X2[1] + X1[1] * X2[1]
"Is $(tree_formula_string(c2)) deterministic? $(isdeterministic(c2))"
```
