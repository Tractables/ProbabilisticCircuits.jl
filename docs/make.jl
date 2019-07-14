using Documenter

include("../src/Circuits/Circuits.jl")
using .Juice

makedocs(
    sitename = "Juice.jl",
    format = Documenter.HTML(),
    doctest = true,
    modules = [Juice]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
