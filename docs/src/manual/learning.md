# [Learning](@id man-learning)

In this section we provide few learning scenarios for circuits. In general, learning tasks for PCs can be separted into two categories: paramter learning and structure learning.


## Learn a Circuit

You can use [`learn_circuit`](@ref) to learn a probabilistic circuit from the data (both paramter and structure learning).


```@example learning
using LogicCircuits
using ProbabilisticCircuits
train_x, valid_x, test_x = twenty_datasets("nltcs")

pc = learn_circuit(train_x; maxiter=100);

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters. " *  
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"
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

## Misc Options

In this sections, we provide options to have more control in learning circuits. For example, what if we only want to do paramter learning.

### Paramter Learning

Given a fixed structure for the PC, the goal of paramter learning is to estimate the parameters so that likelihood is maximized.

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

1. Choose an initial structure and learn paramters
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
