# [Learning PCs](@id man-learning)

In this section we provide few learning scenarios for circuits. In general, learning tasks for PCs can be separted into two categories: paramter learning and structure learning.

### Paramter Learning

Given a fixed structure for the PC and the dataset, the goal of paramter learning is to estimate the parameters so that likelihood is maximized.

First, we load a dataset and initilize a PC with a fully factorized distribution:

```@example learning
using LogicCircuits #hide
using ProbabilisticCircuits #hide
train_x, valid_x, test_x = twenty_datasets("nltcs")
v = Vtree(num_features(train_x), :balanced)
pc = fully_factorized_circuit(StructProbCircuit, v);
"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters." *  
"Train log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```

Given fully observed data, we can do maximum likelihood estimatation (MLE) as follows:

```@example learning
estimate_parameters(pc, train_x; pseudocount=1.0);
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

"PC: $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters." *
"Training set log-likelihood is $(log_likelihood_avg(pc, train_x))"  
```
