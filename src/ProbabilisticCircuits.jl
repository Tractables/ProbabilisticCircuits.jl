module ProbabilisticCircuits

    using DirectedAcyclicGraphs
    import DirectedAcyclicGraphs as DAGs # shorthand

    # reexport from DAGs
    export num_nodes, num_edges

    include("nodes/abstract_nodes.jl")
    
    include("nodes/input_distributions.jl")
    include("nodes/indicator_dist.jl")
    include("nodes/categorical_dist.jl")
    include("nodes/binomial_dist.jl")
    include("nodes/gaussian_dist.jl")

    include("nodes/plain_nodes.jl")

    include("bits_circuit.jl")
    include("traversal.jl")

    include("queries/likelihood.jl")
    include("queries/likelihood_cpu.jl")
    include("queries/map.jl")
    include("queries/map_cpu.jl")
    include("queries/sample.jl")
    include("queries/sample_cpu.jl")
    include("queries/flow.jl")

    include("parameters/em.jl")

    include("io/io.jl")

    include("structures/hclts.jl")
    include("structures/rat.jl")
    
end
