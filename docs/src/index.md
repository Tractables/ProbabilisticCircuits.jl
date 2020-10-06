# ProbabilisticCircuits.jl

This module provides a Julia implementation of Probabilistic Circuits (PCs),  tools to learn structure and parameters of PCs from data, and tools to do tractable exact inference with them. 

### What are Probabilistic Circuits?

Probabilistic Circuits provides a unifying framework for several family of tractable probabilistic models. PCs are represented as a computational graphs that define a joint probability distribution as recursive mixtures
(sum units) and factorizations (product units) of simpler distributions (input units).

Given certain structural properties, PCs enable different range of tractable exact probabilistic queries such as computing marginals, conditionals, maximum a posteriori (MAP), and more advanced probabilistic queries.

In additon to parameters, the structure of PCs can also be learned from data. There are several approaches in learning PCs, while keeping the needed structural constrains intact. Currently, This module includes implementation for few of these approaches with plans to add more over time.

Additionally, parallelism (on both CPU and GPU) is leveraged to provide faster implementation of learning and inference.

### Where to learn more about them?

For an overview of the motivation and theory behind PCs, you can start by watching the ECML-PKDD tutorial on Probabilistic Circuits. 

- Probabilistic Circuits: Representations, Inference, Learning and Theory ([Video](https://www.youtube.com/watch?v=2RAG5-L9R70))

For more details and additional references, you can refer to:

- Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models ([PDF](http://starai.cs.ucla.edu/papers/ProbCirc20.pdf))