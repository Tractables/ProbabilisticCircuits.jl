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

EVI refers to computing the probability when full evidence is given, i.e. when ``x`` is fully observed, the output is ``p(x)``. We can use [`EVI`](@ref) method to compute ``\log{p(x)}``:

```@example queries
probs = EVI(pc, data);
probs[1:3]
```

Computing the [`EVI`](@ref) of a mixture of circuits works the same way. You may either pass weights for the weighted mixture probability, or pass a component index to individually evaluate each component.

```@example queries
mix, mix_weights, _ = learn_strudel(data; num_mix = 10, init_maxiter = 10, em_maxiter = 100)
# This computes the weighted probability
probs = EVI(mix, data, mix_weights);
# Alternatively, we may want to compute the probability of a single component
c_prob = EVI(mix, data; component_idx = 1);
c_prob[1:3]
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

Just like [`EVI`](@ref), [`MAR`](@ref) works the same way for mixtures.

```@example queries
# Full weighted marginal probability
probs_mar = MAR(mix, data, mix_weights);
# Individual component's marginal probability
c_probs_mar = MAR(mix, data; component_idx = 1);
c_probs_mar[1:3]
```

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
data_miss = make_missing(data,keep_prob=0.5);
states, probs = MAP(pc, data_miss);
probs[1:3]
```

## Sampling

We can also sample from the distrubtion ``p(x)`` defined by a Probabilistic Circuit. You can use [`sample`](@ref) to achieve this task.

```@example queries
samples, lls = sample(pc, 100);
lls[1:3]
```

Additionally, we can do conditional samples ``x \sim p(x \mid x^o)``, where ``x^o`` are the observed features (``x^o \subseteq x``), and could be any arbitrary subset of features.

```@example queries
#3 random evidences for the examples
evidence = DataFrame(rand( (missing,true,false), (2, num_variables(pc))))

samples, lls = sample(pc, 3, evidence);
lls
```

## Expected Prediction (EXP)

Expected Prediction (EXP) is the task of taking expectation of a discrimintative model w.r.t a generative model conditioned on evidemce (subset of features observed).

``\mathbb{E}_{x^m \sim p(x^m \mid x^o)} [ f(x^o x^m) ]``

In the case where ``f`` and ``p`` are circuit, and some structural constrains for the pair, we can do this expectation and higher moments tractably. 
You can use [`Expectation`](@ref) and [`Moment`](@ref) to compute the expectations.

```@example queries
using DataFrames

pc = zoo_psdd("insurance.psdd")
rc = zoo_lc("insurance.circuit", 1)

# Using samples from circuit for the example; replace with real data
data, _ = sample(pc, 10);
data = make_missing(DataFrame(data));

exps, exp_cache = Expectation(pc, rc, data)

exps[1:3]
```

```@example queries
second_moments, moment_cache = Moment(pc, rc, data, 2);
exps[1:3]
```

```@example queries
stds = sqrt.( second_moments - exps.^2 );
stds[1:3]
```
