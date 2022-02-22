# [Quick Demo](@id man-demo)

In this section, we provide quick code snippets to get started with ProbabilisticCircuits and provide basic understanding of them. PCs are represented as a computational graphs that define a joint probability distribution as recursive mixtures (sum units) and factorizations (product units) of simpler distributions (input units).

Generally, we learn structure and parameters of circuit from data. Alternatively, we can also specify circuits in code. For example, the following snippet defines a circuit depending on 3 random variables. The `literals` function returns the input units of the circuit, in this case we get 6 different units (3 for positive literals, and 3 for negative literlas).  You can use `*` and `+` operators to build a circuits.

```@example demo
using ProbabilisticCircuits;

X1, X2, X3 = [ProbabilisticCircuits.PlainInputNode(i, Indicator(true)) for i=1:3]
X1_, X2_, X3_ = [ProbabilisticCircuits.PlainInputNode(i, Indicator(false)) for i=1:3]

pc = 0.3 * (X1_ *
             (0.2 * X2_ + 0.8 * X3)) +
     0.7 * (X1 *
             (0.4 * X2 + 0.6 * X3_));

nothing # hide
```

You can ask basic questions about PCs, such as (1) how many variables they depends on, (2) how many nodes, (3) how many edges, (4) or how many parameters they have.

```@example demo
num_randvars(pc)
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


We can also plot circuits using `plot(pc)` to see the computation graph (structure and parameters). The output of `plot(pc)` has a type of `TikzPictures.TikzPicture`. Generally, notebooks automatically renders it and you see the figure in the notebook. 

```@example demo
plot(pc)
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