| Build Status                                                                                                                                                                                                                                                                       	|                                              Documentation                                             	|
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|:------------------------------------------------------------------------------------------------------:	|
| [![Build Status](https://travis-ci.org/Juice-jl/ProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/Juice-jl/ProbabilisticCircuits.jl)	 [![codecov](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl) 	| [![](https://img.shields.io/badge/docs-stable-green.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/dev) 	|



# ProbabilisticCircuits.jl
Probabilistic Circuits - part of Juice (Julia Circuit Empanada)

## Installation

To install the latest stable release, run:

    julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'

To install from the latest commits of the package (recommented to also use latest commits of `LogicCircuits.jl`), run:

    julia -e 'using Pkg; Pkg.add([PackageSpec(url="https://github.com/Juice-jl/LogicCircuits.jl.git"),PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git")])'

This will automatically install all dependencies described in `Project.toml`.
The first time you run `using ProbabilisticCircuits` in Julia, it will precompile the package and all its dependencies.

**Note**: Currently `ProbabilisticCircuits` installation and build fails on Windows due to one of our dependencies (see [Issue 3](https://github.com/Juice-jl/ProbabilisticCircuits.jl/issues/3) for more details). Additionally, on Linux you need to have a C++ compiler installed due to the same dependency (See [BlossomV.jl](https://github.com/mlewe/BlossomV.jl)).

## Testing

To make sure everything is working correctly, you can run our test suite as follows. The first time you run the tests will trigger a few slow downloads of various test resources.

```bash
    julia --color=yes -e 'using Pkg; Pkg.test("ProbabilisticCircuits")'
```

## Development

If you are interested in modifying the package please see the [development README](README_DEV.md).

## Documentation

To build the documentation locally, follow the instructions in the [docs Readme](docs/Readme.md).
