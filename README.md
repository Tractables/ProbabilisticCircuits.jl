[![Build Status](https://travis-ci.com/UCLA-StarAI/Juice.jl.svg?token=WqP1S31vh9msACoVUepf&branch=master)](https://travis-ci.com/UCLA-StarAI/Juice.jl) [![codecov](https://codecov.io/gh/UCLA-StarAI/Juice.jl/branch/master/graph/badge.svg?token=ORgtXXr8Uo)](https://codecov.io/gh/UCLA-StarAI/Juice.jl)

# Juice.jl
Julia Circuit Empanada

# Requirements

Julia 1.3

# Installation

You can use the `Project.toml` that comes with Juice to activate a Julia environment with all dependencies
Concretely, the following command will download and install all required packages.

    julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile();'

You can run the following commands to ensure Julia will find a local version of Juice at `~/code/Juice.jl` and is able to use it on all processors (change `/code/Juice.jl` to be the correct path):
    
    mkdir  -p ~/.julia/config
    echo -e 'using Distributed\n @everywhere push!(LOAD_PATH, "$(homedir())/code/Juice.jl")' > ~/.julia/config/startup.jl

# Documentation

To build the documentation locally, run the following to build the documentation, and then open `docs/build/index.html`.

    julia docs/make.jl

# Troubleshooting

## Installation

If the MLDataSets package fails to build, install the following (Ubuntu):

  ``sudo apt-get install zlib1g-dev libncurses5-dev``

## Testing

To run all the test cases in the `/test` folder, do:

    julia --color=yes -pauto runtests.jl

The flag `-pauto` parallelizes the tests across all CPU cores.
You can also run all the tests for a single (sub-)module, for example:

    julia --color=yes -pauto runtests.jl IO

Or even any individual test, for example:

    julia --color=yes -pauto runtests.jl IO/VtreeParserTest.jl
