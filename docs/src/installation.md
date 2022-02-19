# Installation

### Prerequisites

Julia 1.6 or greater. For installation, please refer to [the official Julia Website](https://julialang.org/downloads/).


### Installing ProbabilisticCircuits

You can use Julia's package manager, Pkg, to install this module and its dependencies. There are different options on how to do that, for example through command line or julia REPL. For more information and options on how to use Julia pacakge manager, please refer to [Pkg's Documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html).


#### From Command Line

To install the latest stable release, run:

```bash
julia -e 'using Pkg; Pkg.add("ProbabilisticCircuits")'
```

You can also install the package with the latest commits on master branch.

```bash
julia -e 'using Pkg; Pkg.add([PackageSpec(url="https://github.com/Juice-jl/ProbabilisticCircuits.jl.git")])'
```
#### From Package mode
!!! note
    To get to Pkg mode, you need to run `julia`, then to press `]`. Press backspace or ^C to get back to normal REPL mode.

While in Pkg mode, run the following to install the latest release:

```julia
add ProbabilisticCircuits
```

Similarly, to install from the latest commits on master branch, run:

```
add ProbabilisticCircuits#master
```
### Testing
If you are installing the latest commit, we recommend running the test suite to make sure everything is in order, to do that run:

```bash
julia --color=yes -e 'using Pkg; Pkg.test("ProbabilisticCircuits")'
```

**Note**: If you want the tests to run faster, you can use multiple cores. To do that set the following environment variable (default = 1 core):

```bash
export JIVE_PROCS=8
```
