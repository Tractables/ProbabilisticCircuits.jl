using Documenter

# include("../src/ProbabilisticCircuits.jl")
using ProbabilisticCircuits

makedocs(
    sitename = "ProbabilisticCircuits.jl Documentation",
    format = Documenter.HTML(prettyurls = false),
    doctest = true,
    modules = [ProbabilisticCircuits],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    target = "build",
    repo = "github.com/Juice-jl/ProbabilisticCircuits.jl.git",
    branch = "gh-pages",
    devbranch = "master",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#"],
)