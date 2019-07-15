using Documenter

include("../src/Juice/Juice.jl")
using .Juice

makedocs(
    sitename = "Juice.jl Documentation",
    format = Documenter.HTML(prettyurls = false),
    doctest = true,
    modules = [Juice],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
