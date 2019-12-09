[![Build Status](https://travis-ci.com/UCLA-StarAI/Juice.jl.svg?token=WqP1S31vh9msACoVUepf&branch=master)](https://travis-ci.com/UCLA-StarAI/Juice.jl) [![codecov](https://codecov.io/gh/UCLA-StarAI/Juice.jl/branch/master/graph/badge.svg?token=ORgtXXr8Uo)](https://codecov.io/gh/UCLA-StarAI/Juice.jl)

# Juice.jl
Julia Circuit Empanada

# Requirements

Julia 1.2

# Documentation

To build the documentation locally, run the following to build the documentation, and then open `docs/build/index.html`.

    julia docs/make.jl

# Developement

## Dependencies

For local developlement, you can use the `Project.toml` to activate a Julia environment with the requirements. See [the documentation](https://julialang.github.io/Pkg.jl/stable/environments/) for more details on how to use environments.

Alternatively, you can run the following to install the dependencies in your default Julia environment (to be deprecated soon).

 ``julia install.jl``


## Using Revise

`Revise.jl` allows you to modify code and use the changes without restarting Julia. For more information refer to [their github repository](https://github.com/timholy/Revise.jl).
 To use the Revise functionality while importing `Juice`, run the following command at the beginning of your REPL session. You may need to add additional files to track in `Debug.jl`.

    using Revise; include("sandbox/Debug.jl");

## Sandbox

The `sandbox` folder will contain some (old) examples on how to use the library.

## Testing

We will be adding automated tests under `/test` folder. To run all the test cases, run:

    julia --color=yes -pauto test/run.jl

The flag `-pauto` parallelizes the tests across all CPU cores.
You can also run all the tests for a single (sub-)module, for example:

    julia --color=yes -pauto test/run.jl test/Juice/IO

Or even any individual test, for example:

    julia --color=yes -pauto test/run.jl test/Juice/IO/VtreeParserTest.jl

Make sure to run the tests before commiting new code.


## Troubleshooting

If MLDataSets package fails to build, mostly likely will be fixed by installing following (Ubuntu):

  ``sudo apt-get install zlib1g-dev libncurses5-dev``
