using Documenter
#using DocumenterLaTeX
using ProbabilisticCircuits
using Literate

#######################################
# 1/ generate the top-level README.md
#######################################

source_dir = "$(@__DIR__)/src"

"replace script includes with file content in Literate code"
function replace_includes(str)
    pat = r"include\(\"(.*)\"\)"
    m = match(pat, str)
    while !isnothing(m)
        str = replace(str, "$(m.match)" =>
                read("$source_dir/$(m[1])", String))
        m = match(pat, str)
    end
    str
end

"hide `#plot` lines in Literate code"
function hide_plots(str)
    str = replace(str, r"#plot (.)*[\n\r]" => "")
    replace(str, r"#!plot (.*)[\n\r]" => s"\g<1>\n")
end

"show `#plot` lines in Literate code"
function show_plots(str)
    str = replace(str, r"#!plot (.)*[\n\r]" => "")
    replace(str, r"#plot (.*)[\n\r]" => s"\g<1>\n")
end

Literate.markdown("$source_dir/README.jl", "$(@__DIR__)/../"; documenter=false, credit=false, execute=true, 
    preprocess = hide_plots ∘ replace_includes)

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

const  pages = [
    "Home" => "index.md",
    "Manual" => [
        "manual/demo.md",
        "manual/queries.md",
        "manual/learning.md",
        "manual/gpu.md"
    ],
    "API" => [
        "api/common.md",
        "api/input_dists.md",
        "api/probabilistic_circuits.md",
        "api/types.md"
    ],
    "Installation" => "installation.md",
];

const format = if ("pdf" in ARGS)
    LaTeX(platform  = "native")
else
   Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://tractables.github.io/ProbabilisticCircuits.jl/stable/",
        assets = ["assets/favicon.ico"],
        analytics = "UA-136089579-2",
        highlights = ["yaml"],
        collapselevel = 1,
    )
end

makedocs(
    sitename = "ProbabilisticCircuits.jl",
    format = format,
    doctest = true,
    modules = [ProbabilisticCircuits],    
    pages = pages,
    linkcheck_ignore = [
        # We'll ignore links that point to GitHub's edit pages, as they redirect to the
        # login screen and cause a warning:
        r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    target = "build",
    repo = "github.com/tractables/ProbabilisticCircuits.jl.git",
    branch = "gh-pages",
    devbranch = "master",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#"],
)