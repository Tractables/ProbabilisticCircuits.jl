[![Build Status](https://travis-ci.org/Juice-jl/ProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/Juice-jl/LogicCircuits.jl)
[![codecov](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Juice-jl/LogicCircuits.jl)

# ProbabilisticCircuits.jl
Probabilistic Circuits - part of Juice (Julia Circuit Empanada)

## Requirements

Julia 1.3

## Installation

You can use the `Project.toml` that comes with ProbabilisticCircuits to activate a Julia environment with all dependencies
Concretely, the following command run from the ProbabilisticCircuits directory will download and install all required packages.

    julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile();'

Suppose you want Julia to use your local copy of the `ProbabilisticCircuits.jl` code, which you stored under `~/Juice/ProbabilisticCircuits.jl/`.
You can run the following commands to ensure Julia will find this code and use it on all processors:
    
    mkdir  -p ~/.julia/config
    echo -e 'using Distributed\n @everywhere push!(LOAD_PATH, "$(homedir())/Juice/ProbabilisticCircuits.jl")' >> ~/.julia/config/startup.jl

## Documentation

To build the documentation locally, run the following to build the documentation, and then open `docs/build/index.html`.

    julia -e 'using Pkg; Pkg.activate("./docs"); Pkg.instantiate(); include("docs/make.jl");

## Troubleshooting

### Testing

To run all the tests:

    julia --color=yes -pauto test/runtests.jl

The flag `-pauto` parallelizes the tests across all CPU cores.
You can also run all the tests for a single (sub-)module, for example:

    julia --color=yes -pauto test/runtests.jl IO

Or even any individual test, for example:

    julia --color=yes -pauto test/runtests.jl IO/CircuitSaverTest.jl
