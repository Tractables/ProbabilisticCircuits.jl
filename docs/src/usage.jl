# ### Quick Tutorial

# Assuming that the ProbabilisticCircuits Julia package has been installed with `julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'`, we can start using it as follows.

using ProbabilisticCircuits

# ### Reasoning with manually constructed circuits

# We begin by creating three positive literals (boolean variables) and manually construct a probabilistic circuit that encodes the following Naive Bayes distribution over them: Pr(rain, rainbow, wet) = Pr(rain) * Pr(rainbow|rain) * Pr(wet|rain).
rain, rainbow, wet = pos_literals(ProbCircuit, 3)
rain_pos = (0.7 * rainbow + 0.3 * (-rainbow)) * (0.9 * wet + 0.1 * (-wet)) # Pr(rainbow|rain=1) * Pr(wet|rain=1)
rain_neg = (0.2 * rainbow + 0.8 * (-rainbow)) * (0.3 * wet + 0.7 * (-wet)) # Pr(rainbow|rain=0) * Pr(wet|rain=0)
circuit = 0.4 * (rain * rain_pos) + 0.6 * ((-rain) * rain_neg); # Pr(rain, rainbow, wet)

# Just like any probability distributions, we can evaluate the probabilistic circuit on various inputs. Note that since log probabilities are used in ProbCircuit for numerical stability, we need to take exponent to get the probabilities.
exp(circuit(true, true, true)) # Pr(rain=1, rainbow=1, wet=1)

exp(circuit(true, false, false)) # Pr(rain=1, rainbow=0, wet=0)

# From the above examples, we see that it is less likely to rain if we do not see rainbows and the streets are not wet.