# ProbabilisticCircuits.jl for Developers

Follow these instructions to install and use ProbabilisticCircuits.jl as a developer of the package.

## Installation

Install the Julia package in development mode by running

    julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git"))'

By default this will install the package at `~/.julia/dev` and allow you to change the code there. See the [Pkg manual](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1) for more details. One can adjust the development directory using environment variables or simply create a symbolic link to/from your favorite development directory.


Depending on your usecase you might also want to have `LogicCircuits.jl` in develop mode, in that case run the following to do both:

    julia -e 'using Pkg; Pkg.develop([PackageSpec(url="https://github.com/Juice-jl/LogicCircuits.jl.git"),PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git")])'


## Testing

Make sure to run the tests before commiting new code.

To run all the tests:

    julia --color=yes -pauto test/runtests.jl

The flag `-pauto` parallelizes the tests across all CPU cores.
You can also run all the tests for a single (sub-)module, for example:

    julia --color=yes -pauto test/runtests.jl IO

Or even any individual test, for example:

    julia --color=yes -pauto test/runtests.jl IO/CircuitSaverTest.jl
    
   
   
   ## Releasing New Versions

Only do this for when the repo is in stable position, and we have decent amount of changes from previous version.

1. As much as possible, make sure to first release a new version for `LogicCircuits.jl`.
2. Bump up the version in `Project.toml`
3. Use [Julia Registrator](https://github.com/JuliaRegistries/Registrator.jl) to submit a pull request to julia's public registry. 
    - The web interface seems to be the easiest. Follow the instructions in the generated pull request and make sure there is no errors. For example [this pull request](https://github.com/JuliaRegistries/General/pull/15350).
3. Github Release. TagBot is enabled for this repo, so after the registrator merges the pull request, TagBot automatically does a github release in sync with the registrar's new version. 
   - Note: TagBot would automatically include all the closed PRs and issues since the previous version in the release note, if you want to exclude some of them, refer to [Julia TagBot docs](https://github.com/JuliaRegistries/TagBot).
