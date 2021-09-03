# [Learning](@id man-learning)

In this section we provide few learning scenarios for circuits. In general, learning tasks for PCs can be separted into two categories: parameter learning and structure learning.

## Learn a Circuit

You can use [`learn_circuit`](@ref) to learn a probabilistic circuit from the data (both parameter and structure learning).

```@example learning
using LogicCircuits
using ProbabilisticCircuits
train_x, valid_x, test_x = twenty_datasets("nltcs")

pc = learn_circuit(train_x; maxiter=100);

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters. " *  
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"
```

## Learning a circuit from missing data

You can use [`learn_circuit_miss`](@ref) to learn a probabilistic circuit from missing data, i.e. some feature could be missing for each data point.

```@example learning
train_x_miss = make_missing_mcar(train_x; keep_prob=0.9)
pc = learn_circuit_miss(train_x_miss; maxiter=100);

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters. " *  
"Train marginal-log-likelihood is $(marginal_log_likelihood_avg(pc, train_x))"
```

## Learn a mixture of circuits

We also support learning mixture of circuits using the Strudel algorithm ([`learn_strudel`](@ref)).

```@example learning
using LogicCircuits
using ProbabilisticCircuits
using Statistics

train_x, valid_x, test_x = twenty_datasets("nltcs")

spc, component_weights, lls = learn_strudel(train_x; num_mix = 10, init_maxiter = 20, em_maxiter = 100);

"SPC: $(num_nodes(spc)) nodes, $(num_parameters(spc)) parameters. " *
"Train log-likelihood is $(mean(lls))"
```

## Learn a circuit from logical constraints and data

There are several ways to learn a probabilistic circuit consistent with logical constraints. Juice currently supports the following:

1. Compilation from an SDD;
2. Compilation from a BDD;
3. Relaxation through [`sample_psdd`](@ref).

### Compilation from an SDD

A circuit (more specifically a PSDD) can be easily constructed from an SDD by simply calling the `compile` function.

Let's assume we have the following CNF

```math
\phi(a,b,c,d)=(a\vee\neg b)\wedge(c\vee\neg d)\wedge(a\vee\neg d)
```

as a [`.cnf` file](https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html):

```
/tmp/example.cnf
---
c Encodes the following CNF: ϕ = (1 ∨ ¬2) ∧ (3 ∨ ¬4) ∧ (1 ∨ ¬4)
c
p cnf 4 3
1 -2 0
3 -4 0
1 -4 0
```

First we construct an SDD from the CNF. Here we sample a random vtree as an example (you might want to learn it from data instead with [`learn_vtree`](@ref)).

```@setup sdd
open("/tmp/example.cnf", "w") with f do write(f,
"""
c Encodes the following: ϕ = (1 ∨ ¬2) ∧ (3 ∨ ¬4) ∧ (1 ∨ ¬4)
c
p cnf 4 3
1 -2 0
3 -4 0
1 -4 0
""") end
```

```@example sdd
using LogicCircuits
using ProbabilisticCircuits
using DataFrames

n = 4 # number of variables
V = Vtree(n, :random)
sdd = compile(SddMgr(V), load_cnf("/tmp/example.cnf"))
pc = compile(StructProbCircuit, sdd)
```

Let's check its support.

```@example sdd
# Matrix with all possible worlds.
M = BitMatrix(undef, 2^n, n)
for i in 1:size(M, 1) M[i,:] .= [c == '0' ? false : true for c in reverse(bitstring(i-1)[end-n-1:end])] end
display(M)

# Evaluate SDD.
display(sdd.(eachrow(M)))
```

And now the probabilities:

```@example sdd
# Evaluate the PSDD support.
EVI(pc, DataFrame(M))
```

### Compilation from a BDD

Compiing from a BDD is straightforward. Let's first create a BDD from the same previous constraints using `LogicCircuits`.

```@example bdd
using LogicCircuits

ϕ = (1 ∨ ¬2) ∧ (3 ∨ ¬4) ∧ (1 ∨ ¬4)
```

Now we either compile the PSDD directly from the BDD and give it random weights:

```@example bdd
using ProbabilisticCircuits, DataFrames

# Get all possible instances with BDD.all_valuations.
M = all_valuations(collect(1:n))
M_D = DataFrame(M)

pc = generate_from_bdd(ϕ, 4)
EVI(pc, M_D)
```

Or compile from a BDD and learn weights from data:

```@example bdd
# Retrieve only possible worlds.
W = M[findall(ϕ.(eachrow(M))),:]
# Assign random probabilities for each world in W.
R = rand(1:20, size(W, 1))
# Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))

pc = learn_bdd(ϕ, D; pseudocount = 0.0)
EVI(pc, M_D)
```

Since BDDs are just right-linear vtree PSDDs, this "compilation" is merely a conversion from BDD
syntax to PC syntax, attributing some weight to edges.

### Sampling a circuit from a relaxation of the constraints

The two previous approaches are effective, but not always adequate. For instance, suppose our data
consists of 6 variables: ``a``, ``b``, ``c``, ``d``, ``e`` and ``f``, where only ``a``, ``b``,
``c`` and ``d`` are constrained (by ``\phi``), and the rest are free. Had we compiled ``\phi`` from
either an SDD or BDD, we'd end up with trivial structures for free variables. For instance, calling
[`learn_bdd`](@ref) (or [`generate_from_bdd`](@ref)) with more variables than the size of the BDD's
scope would result in a fully factorized distribution over the free variables.

To address these issues, we might want to generate a circuit from both free and constrained
variables with [`sample_psdd`](@ref). Unfortunately, to keep the circuit tractable, `sample_psdd`
only provides a relaxation of the constraints.

Let's first encode our constraints as a BDD just like our previous example and make up some random
data.

```@example samplepsdd
using LogicCircuits, DataFrames

n = 6
ϕ = (1 ∨ ¬2) ∧ (3 ∨ ¬4) ∧ (1 ∨ ¬4)
M = all_valuations(collect(1:n))
M_D = DataFrame(M)
W = M[findall(ϕ.(eachrow(M))),:]
R = rand(1:20, size(W, 1))
D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))
```

Now we can sample circuits from ``\phi`` and data ``D``.

```@example samplepsdd
using ProbabilisticCircuits
using LogicCircuits: Vtree

pc = sample_psdd(ϕ, Vtree(n, :random), 16, D)
EVI(pc, M_D)
```

The third argument passed to `sample_psdd` indicates an upper bound on the number of children whose
parents are sum nodes. The higher this bound, the more consistent with ``\phi``.

In situations where background knowledge is not available, we may pass ``\top`` to `sample_psdd` to
only learn from data.

```@example samplepsdd
pc = sample_psdd(⊤, Vtree(n, :random), 16, D)
EVI(pc, M_D)
```

## Learning an ensemble of circuits

[`learn_strudel`](@ref) let's us learn an ensemble of circuits that share the same structure. For
learning ensembles whose components have different structures, we have to use `Ensemble`.

### Ensemble of `sample_psdd`s

We can learn an ensemble of random circuits through `ensemble_sample_psdd`.

```@example samplepsdd
E = ensemble_sample_psdd(10, ϕ, 16, D; strategy = :em)
EVI(E, M_D)
```

Here we used EM to learn the weights of the ensemble. Alternatives are likelihood weighting
(`:likelihood`), uniform weights (`:uniform`) or stacking (`:stacking`).

## Misc Options

In this sections, we provide options to have more control in learning circuits. For example, what if we only want to do parameter learning.

### Parameter Learning

Given a fixed structure for the PC, the goal of parameter learning is to estimate the parameters so that likelihood is maximized.

First, initliaze PC structure with a balanced vtree represneting a fully factorized distribution:

```@example learning
v = Vtree(num_features(train_x), :balanced)
pc = fully_factorized_circuit(StructProbCircuit, v);

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters." *  
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```

No parmater learning is done yet, now let's, do maximum likelihood estimatation (MLE) using [`estimate_parameters`](@ref):

```@example learning
estimate_parameters(pc, train_x; pseudocount=1.0);

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters." *
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```

As we see the likelihood improved, however we are still using a fully factorized distribution. There is room for improvement. For example, we can choose initial structure based on Chow-Liu Trees.

```@example learning
pc, vtree = learn_chow_liu_tree_circuit(train_x)

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters." *
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```

### Structure Learning

There are several different approaches in structure learning. Currently we support the following approach:

1. Choose an initial structure and learn parameters
2. Perform Greedy search for a bigger and better structure by doing operations such as split and clone.
3. Repeat step 2 until satisfied or time limit reached

We start with the Chow-Liu structure we learned in last section, and run few structure learning iterations (20):

```@example learning
pc, vtree = learn_chow_liu_tree_circuit(train_x)
loss(circuit) = ProbabilisticCircuits.heuristic_loss(circuit, train_x)
pc = struct_learn(pc;  
    primitives=[split_step],  
    kwargs=Dict(split_step=>(loss=loss,)),
    maxiter=20)
estimate_parameters(pc, train_x; pseudocount=1.0)

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters. " *
"Training set log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```
