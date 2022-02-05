#src Generate README.md by running `using Literate; Literate.markdown("docs/src/README.jl", "."; documenter=false, credit=false, execute=true)` 

# <img align="right" width="180px" src="https://avatars.githubusercontent.com/u/58918144?s=200&v=4">

# <!-- DO NOT EDIT README.md directly, instead edit docs/README.jl and generate the markdown-->

# # Probabilistic<wbr>Circuits<wbr>.jl

# [![Unit Tests](https://github.com/Juice-jl/ProbabilisticCircuits.jl/workflows/Unit%20Tests/badge.svg)](https://github.com/Juice-jl/ProbabilisticCircuits.jl/actions?query=workflow%3A%22Unit+Tests%22+branch%3Amaster)  [![codecov](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Juice-jl/ProbabilisticCircuits.jl) [![](https://img.shields.io/badge/docs-stable-green.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juice-jl.github.io/ProbabilisticCircuits.jl/dev)

# This package provides functionalities for learning/constructing probabilistic circuits and using them to compute various probabilistic queries. It is part of the [Juice package](https://github.com/Juice-jl) (Julia Circuit Empanada).

# ## Example usage

include("usage.jl")

# ## Testing

# To make sure everything is working correctly, you can run our test suite as follows. The first time you run the tests will trigger a few slow downloads of various test resources.

# ```bash
# julia --color=yes -e 'using Pkg; Pkg.test("ProbabilisticCircuits")'
# ```
