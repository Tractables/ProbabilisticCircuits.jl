# Assuming that the ProbabilisticCircuits Julia package has been installed with `julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'`, we can start using it as follows. We also need [LogicCircuits](https://github.com/Juice-jl/LogicCircuits.jl) for some basic functionalities.

using LogicCircuits, ProbabilisticCircuits

# ### Reasoning with manually constructed circuits

# We begin by creating three positive literals (boolean variables) and manually construct a probabilistic circuit that encodes the following Naive Bayes (NB) distribution over them: Pr(rain, rainbow, wet) = Pr(rain) * Pr(rainbow|rain) * Pr(wet|rain).
rain, rainbow, wet = pos_literals(ProbCircuit, 3)
rain_pos = (0.7 * rainbow + 0.3 * (-rainbow)) * (0.9 * wet + 0.1 * (-wet)) # Pr(rainbow|rain=1) * Pr(wet|rain=1)
rain_neg = (0.2 * rainbow + 0.8 * (-rainbow)) * (0.3 * wet + 0.7 * (-wet)) # Pr(rainbow|rain=0) * Pr(wet|rain=0)
circuit = 0.4 * (rain * rain_pos) + 0.6 * ((-rain) * rain_neg); # Pr(rain, rainbow, wet)

# Just like any probability distributions, we can evaluate the probabilistic circuit on various inputs. Note that since log probabilities are used in ProbCircuit for numerical stability, we need to take exponent to get the probabilities.
exp(circuit(true, true, true)) # Pr(rain=1, rainbow=1, wet=1)
#-
exp(circuit(true, false, false)) # Pr(rain=1, rainbow=0, wet=0)

# From the above examples, we see that it is less likely to rain if we do not see rainbows and the streets are not wet.

# The purpose of this package is to offer a unified tool for efficient learning and inference (i.e., answering probabilistic queries such as marginals and MAP) over probabilistic circuits, which represent a large class of tractable probabilistic models. We first use the above manually constructed circuit to demonstrate several queries that can be answered efficiently. Similar to [logic circuits](https://github.com/Juice-jl/LogicCircuits.jl), answering the following questions requre *decomposability* and *determinism*, which is already satisfied by construction:
isdecomposable(circuit) && isdeterministic(circuit)

# Decomposability allows us to compute marginal probabilities given partial evidence efficiently (linear time w.r.t. the circuit size). For example, we want to ask the probability of observing rainbows. That is, we want to marginalize out the variables rain and wet. This can be done by evaluating the circuit with partial evidence:
exp(circuit(missing, true, missing)) # Pr(rainbow=1)

# Being able to compute marginals immediately offers the ability to compute conditional probabilities. For example, to compute the probability of raining given rainbow=1 and wet=1, we simply take the quotient of Pr(rain=1, rainbow=1, wet=1) and Pr(rainbow=1, wet=1):
exp(circuit(true, true, true) - circuit(missing, true, true)) # Pr(rain=1|rainbow=1, wet=1)

# If we are additionally supplied with the structural property *determinism*, we can answer some more advanced queries. For example, we want to compute the maximum a posteriori (MAP) 