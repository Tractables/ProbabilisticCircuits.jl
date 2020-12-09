## Building Docs

To locally build the docs, run the following commands from root of the repository to Instantiate the docs environment and build the docs.

```bash
julia -e 'using Pkg; Pkg.activate("./docs"); Pkg.instantiate(); include("./docs/make.jl");'
```

The build results will be stored under `docs/build`.

Alternatively, if you have `ProbabilisticCircuits` in development mode and have already instantiated the docs environment, you can simply run the following.

```bash
julia --project=docs docs/make.jl
```

Note that if you do not have the package in the development mode, the docs build would most likely ignore the uncommited changes.


#### Note about Pretty URLs
For easier navigation for local builds its easier to disable pretty URLs. To disable pretty urls run the following instead:

```bash
julia --project=docs docs/make.jl local
```

For more information about pretty URLs, check out  [the documentation](https://juliadocs.github.io/Documenter.jl/stable/man/guide/) for `Documenter.jl`.


## Setting Up LaTeX

Some of plotting tools we use need LaTeX to be installed. Follow the instructions from [TikzPictures.jl](https://github.com/JuliaTeX/TikzPictures.jl) to see what packages are needed. Or you can check the [ci.yml](https://github.com/Juice-jl/ProbabilisticCircuits.jl/blob/master/.github/workflows/ci.yml) which builds our documentation on github actions. If you ran into any issues might be due to old version of TexLive.
