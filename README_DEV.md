# ProbabilisticCircuits.jl for Developers

Follow these instructions to install and use ProbabilisticCircuits.jl as a developer of the package.

## Installation

Install the Julia package in development mode by running

    julia -e 'using Pkg; Pkg.develop(PackageSpec(url="git@github.com:Juice-jl/ProbabilisticCircuits.jl.git"))'

By default this will install the package at `~/.julia/dev` and allow you to change the code there. See the [Pkg manual](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1) for more details. One can adjust the development directory using environment variables or simply create a symbolic link to/from your favorite development directory.

## Testing

Make sure to run the tests before commiting new code.

To run all the tests:

    julia --color=yes -pauto test/runtests.jl

The flag `-pauto` parallelizes the tests across all CPU cores.
You can also run all the tests for a single (sub-)module, for example:

    julia --color=yes -pauto test/runtests.jl IO

Or even any individual test, for example:

    julia --color=yes -pauto test/runtests.jl IO/CircuitSaverTest.jl

## Troubleshooting

When running tests as above, Julia somehow does not install the Jive package even though it is specified in `Project.toml`. 
Add it manually:

    julia -e 'using Pkg; Pkg.add("Jive")'
