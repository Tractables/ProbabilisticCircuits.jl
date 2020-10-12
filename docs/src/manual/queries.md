# [Queries](@id man-queries)

In this section, we go over most common probabilistic reasoning tasks, and provide code snippets to compute those queries. 

### Setup
First, we load some pretrained PC, and the corresponding data.

```@setup queries
# This is needed to hide output from downloading artifacts
using LogicCircuits # hide
using ProbabilisticCircuits; #hide
pc = zoo_psdd("plants.psdd")
data, _, _ = twenty_datasets("plants");
```

```@example queries
using LogicCircuits # hide
using ProbabilisticCircuits; #hide
pc = zoo_psdd("plants.psdd")
data, _, _ = twenty_datasets("plants");
println("circuit with $(num_nodes(pc)) nodes and $(num_parameters(pc)) parameters.")
println("dataset with $(num_features(data)) features and $(num_examples(data)) examples.")
```

## Full Evidence (EVI)

EVI refers to computing the probability when full evidence is given, i.e. when ``x`` is fully oberserved, the output is ``p(x)``. We can use [`EVI`](@ref) method to compute ``\log{p(x)}``:

```@example queries
probs = EVI(pc, data);
probs[1:3]
```


## Partial Evidence (MAR)

In this case we have some missing values. Let ``x^o`` denote the observed features, and ``x^m`` the missing features. We would like to compute ``p(x^o)`` which is defined as ``p(x^o) = \sum_{x^m} p(x^o, x^m)``. Of course, computing this directly by summing over all possible ways to fill the missing values is not tractable. 

The good news is that given a **smooth** and **decomposable** PC, the marginal can be computed exactly and in linear time to the size of the PC.


First, we randomly make some features go `missing`:

```@example queries
using DataFrames
function make_missing(d::DataFrame;keep_prob=0.8)    
    m = missings(Bool, num_examples(d), num_features(d)) 
    flag = rand(num_examples(d), num_features(d)) .<= keep_prob
    m[flag] .= Matrix(d)[flag] 
    DataFrame(m) 
end; 
data_miss = make_missing(data[1:1000,:]);
nothing #hide
```

Now, we can use [`MAR`](@ref) to compute the marginal queries:

```@example queries
probs = MAR(pc, data_miss);
probs[1:3]
```

Note that [`MAR`](@ref) can also be used to compute probabilisties even if all data is observed, in fact it should give the same results as [`EVI`](@ref). However, if we know all features are observed, we suggest using EVI as its faster in general.

```@example queries
probs_mar = MAR(pc, data);
probs_evi = EVI(pc, data);

probs_mar ≈ probs_evi
```


## Conditionals (CON)

In this case, given observed features ``x^o``, we would like to compute ``p(Q \mid x^o)``, where ``Q`` is a subset of features disjoint with ``x^o``. We can leverage Bayes rule to compute conditionals as two seperate MAR queries as follows:

```math
p(q \mid x^o) = \cfrac{p(q, x^o)}{p(x^o)}
```

Currently, this has to be done manually by the user. We plan to add a simple API for this case in the future.

## Maximum a posteriori (MAP, MPE)

In this case, given the observed features ``x^o`` the goal is to fill out the missing features in a way that ``p(x^m, x^o)`` is maximized.


We can use the [`MAP`](@ref) method to compute MAP, which outputs the states that maximize the probability and returns the probabilities themselves.

```@example queries
data_miss = make_missing(data,keep_prob=0.5);
states, probs = MAP(pc, data_miss);
probs[1:3]
```

## Probability of logical Events

## Expected Prediction

## Same Decision Probability
