using Documenter
using ProbabilisticCircuits

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end


makedocs(
    sitename = "ProbabilisticCircuits.jl",
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://juice-jl.github.io/ProbabilisticCircuits.jl/stable/",
        assets = ["assets/favicon.ico"],
        analytics = "UA-136089579-2",
        highlights = ["yaml"],
    ),
    doctest = true,
    modules = [ProbabilisticCircuits],
    linkcheck_ignore = [
        # We'll ignore links that point to GitHub's edit pages, as they redirect to the
        # login screen and cause a warning:
        r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)"
    ], 
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Installation" => "manual/installation.md",
            "Examples" => "manual/examples.md"
        ],
        "API" => Any[
            "Public" => "api/public.md",
            "Internals" => map(
                s -> "api/internals/$(s)",
                sort(readdir(joinpath(@__DIR__, "src/api/internals")))
            ),
        ],
    ],
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