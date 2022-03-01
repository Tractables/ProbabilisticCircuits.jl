# [Queries](@id man-queries)

In this section, we go over most common probabilistic reasoning tasks, and provide code snippets to compute those queries.

### Setup
First, we load some pretrained PC, and the corresponding data.

```@setup queries
# This is needed to hide output from downloading artifacts
using CircuitModelZoo; #hide
using ProbabilisticCircuits; #hide
using DensityEstimationDatasets; #hide
pc = read(zoo_psdd_file("plants.psdd"), ProbCircuit);
data, _, _ = twenty_datasets("plants");
```

```@example queries
using CircuitModelZoo: zoo_psdd_file
using DensityEstimationDatasets: twenty_datasets
using ProbabilisticCircuits
using Tables

pc = read(zoo_psdd_file("plants.psdd"), ProbCircuit);
data, _, _ = twenty_datasets("plants");
data = Tables.matrix(data);
println("circuit with $(num_nodes(pc)) nodes and $(num_parameters(pc)) parameters.")
println("dataset with $(size(data, 2)) features and $(size(data, 1)) examples.")
```

## Full Evidence (EVI)

EVI refers to computing the probability when full evidence is given, i.e. when ``x`` is fully observed, the output is ``p(x)``. We can use [`loglikelihoods`](@ref) method to compute ``\log{p(x)}``:

```@example queries
probs = loglikelihoods(pc, data[1:100, :]; batch_size=64);
probs[1:3]
```

## Partial Evidence (MAR)

In this case we have some missing values. Let ``x^o`` denote the observed features, and ``x^m`` the missing features. We would like to compute ``p(x^o)`` which is defined as ``p(x^o) = \sum_{x^m} p(x^o, x^m)``. Of course, computing this directly by summing over all possible ways to fill the missing values is not tractable. 

The good news is that given a **smooth** and **decomposable** PC, the marginal can be computed exactly and in linear time to the size of the PC.

First, we randomly make some features go `missing`.

```@example queries
using DataFrames
using Tables
function make_missing(d; keep_prob=0.8)
    m = missings(Bool, size(d)...)
    flag = rand(size(d)...) .<= keep_prob
    m[flag] .= d[flag]
    return m
end;
data_miss = make_missing(data[1:1000,:]);
nothing #hide
```

Now, we can use [`loglikelihoods`](@ref) to compute the marginal queries.

```@example queries
probs = loglikelihoods(pc, data_miss; batch_size=64);
probs[1:3]
```

Note that [`loglikelihoods`](@ref) can also be used to compute probabilisties if all data is observed, as we saw in previous section.


## Conditionals (CON)

In this case, given observed features ``x^o``, we would like to compute ``p(Q \mid x^o)``, where ``Q`` is a subset of features disjoint with ``x^o``. 
We can use Bayes rule to compute conditionals as two seperate MAR queries as follows:

```math
p(q \mid x^o) = \cfrac{p(q, x^o)}{p(x^o)}
```

Currently, this has to be done manually by the user. We plan to add a simple API for this case in the future.

## Maximum a posteriori (MAP, MPE)

In this case, given the observed features ``x^o`` the goal is to fill out the missing features in a way that ``p(x^m, x^o)`` is maximized.

We can use the [`MAP`](@ref) method to compute MAP, which outputs the states that maximize the probability and the log-likelihoods of each state.

```@example queries
data_miss = make_missing(data[1:1000,:], keep_prob=0.5);
states = MAP(pc, data_miss; batch_size = 64);
size(states)
```

## Sampling

We can also sample from the distrubtion ``p(x)`` defined by a Probabilistic Circuit. You can use [`sample`](@ref) to achieve this task.

```@example queries
samples = sample(pc, 100, [Bool]);
size(samples)
```

Additionally, we can do conditional samples ``x \sim p(x \mid x^o)``, where ``x^o`` are the observed features (``x^o \subseteq x``), and could be any arbitrary subset of features.

```@example queries
#3 random evidences for the examples
evidence = rand( (missing,true,false), (2, num_randvars(pc)));

samples = sample(pc, 3, evidence; batch_size = 2);
size(samples)
```
