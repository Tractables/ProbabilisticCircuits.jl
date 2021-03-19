<img align="right" width="180px" src="https://avatars.githubusercontent.com/u/58918144?s=200&v=4">

<!-- DO NOT EDIT README.md directly, instead edit docs/README.jl and generate the markdown-->

# Probabilistic<wbr>Circuits<wbr>.jl

[![Unit Tests](https://github.com/Juice-jl/ProbabilisticCircuits.jl/workflows/Unit%20Tests/badge.svg)](https://github.com/Juice-jl/ProbabilisticCircuits.jl/actions?query=workflow%3A%22Unit+Tests%22+branch%3Amaster)  [![codecov](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl) [![](https://img.shields.io/badge/docs-stable-green.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/dev)

## Example usage

Assuming that the ProbabilisticCircuits Julia package has been installed with `julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'`, we can start using it as follows. We also need [LogicCircuits](https://github.com/Juice-jl/LogicCircuits.jl) for some basic functionalities.

```julia
using LogicCircuits, ProbabilisticCircuits
```

### Reasoning with manually constructed circuits

We begin by creating three positive literals (boolean variables) and manually construct a probabilistic circuit that encodes the following Naive Bayes (NB) distribution over them: Pr(rain, rainbow, wet) = Pr(rain) * Pr(rainbow|rain) * Pr(wet|rain).

```julia
rain, rainbow, wet = pos_literals(ProbCircuit, 3)
rain_pos = (0.7 * rainbow + 0.3 * (-rainbow)) * (0.9 * wet + 0.1 * (-wet)) # Pr(rainbow|rain=1) * Pr(wet|rain=1)
rain_neg = (0.2 * rainbow + 0.8 * (-rainbow)) * (0.3 * wet + 0.7 * (-wet)) # Pr(rainbow|rain=0) * Pr(wet|rain=0)
circuit = 0.4 * (rain * rain_pos) + 0.6 * ((-rain) * rain_neg); # Pr(rain, rainbow, wet)
```

Just like any probability distributions, we can evaluate the probabilistic circuit on various inputs. Note that since log probabilities are used in ProbCircuit for numerical stability, we need to take exponent to get the probabilities.

```julia
exp(circuit(true, true, true)) # Pr(rain=1, rainbow=1, wet=1)
```

```
0.252f0
```

```julia
exp(circuit(true, false, false)) # Pr(rain=1, rainbow=0, wet=0)
```

```
0.011999999f0
```

From the above examples, we see that it is less likely to rain if we do not see rainbows and the streets are not wet.

The purpose of this package is to offer a unified tool for efficient learning and inference (i.e., answering probabilistic queries such as marginals and MAP) over probabilistic circuits, which represent a large class of tractable probabilistic models. We first use the above manually constructed circuit to demonstrate several queries that can be answered efficiently. Similar to [logic circuits](https://github.com/Juice-jl/LogicCircuits.jl), answering the following queries require *decomposability* and *determinism*, which is already satisfied by construction:

```julia
isdecomposable(circuit) && isdeterministic(circuit)
```

```
true
```

Decomposability allows us to compute marginal probabilities given partial evidence efficiently (linear time w.r.t. the circuit size). For example, we want to ask the probability of observing rainbows. That is, we want to marginalize out the variables rain and wet. This can be done by evaluating the circuit with partial evidence:

```julia
exp(circuit(missing, true, missing)) # Pr(rainbow=1)
```

```
0.39999998f0
```

Being able to compute marginals immediately offers the ability to compute conditional probabilities. For example, to compute the probability of raining given rainbow=1 and wet=1, we simply take the quotient of Pr(rain=1, rainbow=1, wet=1) and Pr(rainbow=1, wet=1):

```julia
exp(circuit(true, true, true) - circuit(missing, true, true)) # Pr(rain=1|rainbow=1, wet=1)
```

```
0.87500006f0
```

If we are additionally supplied with the structural property *determinism*, we can answer some more advanced queries. For example, we want to compute the maximum a posteriori (MAP) query of the distribution:

```julia
assignments, log_prob = MAP(circuit, [missing, missing, missing])
print("The MAP assignment of the circuit is (rain=$(assignments[1]), rainbow=$(assignments[2]), wet=$(assignments[3])), with probability $(exp(log_prob)).")
```

```
The MAP assignment of the circuit is (rain=false, rainbow=false, wet=false), with probability 0.336.
```

### Building complex circuit structures

### Learning probabilistic circuits from data

ProbabilisticCircuits.jl offers various parameter learning and structure learning algorithms. It further support mini-batch learning on both CPUs and GPUs, which makes learning large models from large datasets very efficient.

We use the binarized MNIST dataset to demonstrate example probabilistic circuit learning functionalities.

```julia
train_data, valid_data, test_data = twenty_datasets("binarized_mnist");
```

We start with learning the parameters of a *decomposable* and *deterministic* probabilistic circuit. We first load the structure of the circuit from file:

```julia
circuit = zoo_psdd("mnist.psdd")
print("The loaded circuit contains $(num_edges(circuit)) edges and $(num_parameters(circuit)) parameters.")
```

```
The loaded circuit contains 11280 edges and 5364 parameters.
```

```julia
print("Structural properties of the circuit: decomposability: $(isdecomposable(circuit)), determinism: $(isdeterministic(circuit)).")
```

```
Structural properties of the circuit: decomposability: true, determinism: true.
```

Given that the circuit is decomposable and deterministic, the maximum likelihood estimation (MLE) of its parameters is in closed-form. That is, we can learn the MLE parameters deterministically:

```julia
t = @elapsed estimate_parameters(circuit, train_data; pseudocount = 0.1)
print("Learning the parameters on a CPU took $(t) seconds.")
```

```
Learning the parameters on a CPU took 0.231752347 seconds.
```

Optionally, we can use GPUs to speedup the learning process:

```julia
t = @elapsed estimate_parameters(circuit, train_data; pseudocount = 0.1, use_gpu = true)
print("Learning the parameters on a GPU took $(t) seconds.")
```

```
Learning the parameters on a GPU took 0.163713883 seconds.
```

Note that the insignificant speedup is due to the fact that the circuit is too small to make full use of the GPU. For large circuits the speedup could be at least ~10x.

After the learning process, we can evaluate the model on the validation/test dataset. Here we use average log-likelihood per sample as the metric (we again utilize GPUs for efficiency):

```julia
avg_ll = log_likelihood_avg(circuit, test_data; use_gpu = true)
print("The average test data log-likelihood is $(avg_ll).")
```

```
The average test data log-likelihood is -137.59309172113925.
```

Besides `estimate_parameters`, ProbabilisticCircuits.jl offers iterative parameter learning algorithms such as Expectation-Maximization (EM) (i.e., `estimate_parameters_em`) and Stochastic Gradient Descent (SGD) (i.e., `sgd_parameter_learning`).

ProbabilisticCircuits.jl also offers functionalities for learning the circuit structure and parameters simultaneously. For example, the Strudel structure learning algorithm is implemented natively in the package, and can be used with a few lines of code:

```julia
circuit_strudel = learn_circuit(train_data; maxiter = 100, verbose = false)
avg_ll = log_likelihood_avg(circuit_strudel, test_data; use_gpu = true)
print("The learned circuit contains $(num_edges(circuit)) edges and $(num_parameters(circuit)) parameters.\n")
print("The average test data log-likelihood is $(avg_ll).")
```

```
The learned circuit contains 11280 edges and 5364 parameters.
The average test data log-likelihood is -134.98600316031514.
```

## Installation

To install the latest stable release, run:

```julia
julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'
```

To install from the latest commits of the package (recommented to also use latest commits of `LogicCircuits.jl`), run:

```julia
julia -e 'using Pkg; Pkg.add([PackageSpec(url="https://github.com/Juice-jl/LogicCircuits.jl.git"),PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git")])'
```

This will automatically install all dependencies described in `Project.toml`.
The first time you run `using ProbabilisticCircuits` in Julia, it will precompile the package and all its dependencies.

**Note**: Currently `ProbabilisticCircuits` installation and build fails on Windows due to one of our dependencies (see [Issue 3](https://github.com/Juice-jl/ProbabilisticCircuits.jl/issues/3) for more details). Additionally, on Linux you need to have a C++ compiler installed due to the same dependency (See [BlossomV.jl](https://github.com/mlewe/BlossomV.jl)).

## Testing

To make sure everything is working correctly, you can run our test suite as follows. The first time you run the tests will trigger a few slow downloads of various test resources.

```bash
julia --color=yes -e 'using Pkg; Pkg.test("ProbabilisticCircuits")'
```

