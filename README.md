[![Build Status](https://travis-ci.org/Juice-jl/ProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/Juice-jl/ProbabilisticCircuits.jl)
[![codecov](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl)

# ProbabilisticCircuits.jl
Probabilistic Circuits - part of Juice (Julia Circuit Empanada)

## Installation

Install the Julia package (and its dependency on `LogicCircuits.jl`) by running

    julia -e 'using Pkg; Pkg.add([PackageSpec(url="https://github.com/Juice-jl/LogicCircuits.jl.git"),PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git")])'

This will automatically install all dependencies described in `Project.toml`.
The first time you run `using ProbabilisticCircuits` in Julia, it will precompile the package and all its dependencies.

To make sure everything is working correctly, you can run our test suite as follows. The first time you run the tests will trigger a few slow downloads of various test resources.

    julia --color=yes -e 'using Pkg; Pkg.test("ProbabilisticCircuits")'

## Development

If you are interested in modifying the package please see the [development README](README_DEV.md).

## Documentation

To build the documentation locally, run the following and then open `docs/build/index.html`.

    julia -e 'using Pkg; Pkg.activate("./docs"); Pkg.instantiate(); include("docs/make.jl");