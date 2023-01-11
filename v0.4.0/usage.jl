using CUDA #hide

# Assuming that the ProbabilisticCircuits Julia package has been installed with `julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'`, we can start using it as follows.

using ProbabilisticCircuits

# ### Reasoning with manually constructed circuits

# We begin by creating three positive literals (boolean variables) and manually construct a probabilistic circuit that encodes a Naive Bayes (NB) distribution with the following form: `Pr(rain, rainbow, wet) = Pr(rain) * Pr(rainbow|rain) * Pr(wet|rain)`.

rain, rainbow, wet = [ProbabilisticCircuits.PlainInputNode(i, Indicator(true)) for i=1:3]
rain_, rainbow_, wet_ = [ProbabilisticCircuits.PlainInputNode(i, Indicator(false)) for i=1:3]

rain_pos = (0.7 * rainbow + 0.3 * (rainbow_)) * (0.9 * wet + 0.1 * (wet_)) # Pr(rainbow|rain=1) * Pr(wet|rain=1)
rain_neg = (0.2 * rainbow + 0.8 * (rainbow_)) * (0.3 * wet + 0.7 * (wet_)) # Pr(rainbow|rain=0) * Pr(wet|rain=0)
circuit = 0.4 * (rain * rain_pos) + 0.6 * ((rain_) * rain_neg); # Pr(rain, rainbow, wet)

# Just like any probability distribution, we can evaluate the probabilistic circuit on various inputs. Note that since log probabilities are used in probabilistic circuits for numerical stability, we need to take exponent of the evaluation output to get the probabilities.
exp(loglikelihoods(circuit, [[true, true, true]]; batch_size = 1)) # Pr(rain=1, rainbow=1, wet=1)
#-
exp(loglikelihoods(circuit, [[true, false, false]]; batch_size = 1)) # Pr(rain=1, rainbow=0, wet=0)

# From the above examples, we see that it is less likely to rain if we do not see rainbows and the streets are not wet.

# Decomposability allows us to compute marginal probabilities given partial evidence efficiently (linear time w.r.t. the circuit size). For example, we want to ask the probability of observing rainbows. That is, we want to marginalize out the variables rain and wet. This can be done by evaluating the circuit with partial evidence:

exp(loglikelihoods(circuit, [[missing, true, missing]]; batch_size = 1))

# Being able to compute marginals immediately offers the ability to compute conditional probabilities. For example, to compute the probability of raining given rainbow=1 and wet=1, we simply take the quotient of Pr(rain=1, rainbow=1, wet=1) and Pr(rainbow=1, wet=1):
exp(circuit(true, true, true) - circuit(missing, true, true)) # Pr(rain=1|rainbow=1, wet=1)

# If we are additionally supplied with the structural property *determinism*, we can answer some more advanced queries. For example, we can to compute the maximum a posteriori (MAP) query of the distribution:
assignments, log_prob = MAP(circuit, [missing, missing, missing]; batch_size=1)
print("The MAP assignment of the circuit is (rain=$(assignments[1]), rainbow=$(assignments[2]), wet=$(assignments[3])), with probability $(exp(log_prob)).")

# Besides the above examples, ProbabilisticCircuits.jl provides functionalities for a wide variety of queries, which are detailed in [this manual](https://juice-jl.github.io/ProbabilisticCircuits.jl/stable/manual/queries/).

# ### Building complex circuit structures

# ProbabilisticCircuits.jl provides tools to compile classic Probabilistic Graphical Models (PGMs) and Tractable Probabilistic Models (TPMs) into probabilistic circuits efficiently. For example, we can compile a factor graph (FG) into a probabilistic circuit with one line of code:
fg = fromUAI(zoo_fg_file("asia.uai")) # Load example factor graph
fg_circuit = ProbCircuit(compile_factor_graph(fg)[1]) # Compile the FG to a PC
print("`fg_circuit` contains $(num_edges(fg_circuit)) edges and $(num_parameters(fg_circuit)) parameters.")

# ### Learning probabilistic circuits from data

# ProbabilisticCircuits.jl offers various parameter learning and structure learning algorithms. It further support mini-batch learning on both CPUs and GPUs, which makes learning large models from large datasets very efficient.

# We use the binarized MNIST dataset to demonstrate example probabilistic circuit learning functionalities.
train_data, valid_data, test_data = twenty_datasets("binarized_mnist");

# We start with learning the parameters of a *decomposable* and *deterministic* probabilistic circuit. We first load the structure of the circuit from file:
circuit = zoo_psdd("mnist.psdd")
print("The loaded circuit contains $(num_edges(circuit)) edges and $(num_parameters(circuit)) parameters.")
#-
print("Structural properties of the circuit: decomposability: $(isdecomposable(circuit)), determinism: $(isdeterministic(circuit)).")

# Given that the circuit is decomposable and deterministic, the maximum likelihood estimation (MLE) of its parameters is in closed-form. That is, we can learn the MLE parameters deterministically:
estimate_parameters!(circuit, train_data; pseudocount = 0.1) #hide
t = @elapsed estimate_parameters!(circuit, train_data; pseudocount = 0.1)
print("Learning the parameters on a CPU took $(t) seconds.")

# Optionally, we can use GPUs to speedup the learning process:
if CUDA.functional() #hide
train_data = to_gpu(train_data)
estimate_parameters!(circuit, train_data; pseudocount = 0.1) #hide
t = @elapsed estimate_parameters!(circuit, train_data; pseudocount = 0.1)
print("Learning the parameters on a GPU took $(t) seconds.")
end #hide

# Note that the insignificant speedup is due to the fact that the circuit is too small to make full use of the GPU. For large circuits the speedup could be at least ~10x.

# After the learning process, we can evaluate the model on the validation/test dataset. Here we use average log-likelihood per sample as the metric (we again utilize GPUs for efficiency):
avg_ll = log_likelihood_avg(circuit, test_data)
print("The average test data log-likelihood is $(avg_ll).")

# Besides `estimate_parameters`, ProbabilisticCircuits.jl offers iterative parameter learning algorithms such as Expectation-Maximization (EM) (i.e., `estimate_parameters_em!`) and Stochastic Gradient Descent (SGD) (i.e., `estimate_parameters_sgd!`).

# ProbabilisticCircuits.jl also offers functionalities for learning the circuit structure and parameters simultaneously. For example, the Strudel structure learning algorithm is implemented natively in the package, and can be used with a few lines of code:
circuit_strudel = learn_circuit(train_data; maxiter = 100, verbose = false)
avg_ll = log_likelihood_avg(circuit_strudel, test_data)
print("The learned circuit contains $(num_edges(circuit)) edges and $(num_parameters(circuit)) parameters.\n")
print("The average test data log-likelihood is $(avg_ll).")