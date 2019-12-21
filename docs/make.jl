using Documenter

# include("../src/ProbabilisticCircuits.jl")
using ProbabilisticCircuits

makedocs(
    sitename = "ProbabilisticCircuits.jl Documentation",
    format = Documenter.HTML(prettyurls = false),
    doctest = true,
    modules = [ProbabilisticCircuits, Data, Utils],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
