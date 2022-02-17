module ProbabilisticCircuits

    using DirectedAcyclicGraphs
    import DirectedAcyclicGraphs as DAGs # shorthand

    # reexport from DAGs
    export num_nodes, num_edges

    include("abstract_nodes.jl")
    
    include("input_distributions.jl")
    include("plain_nodes.jl")
    include("bits_circuit.jl")
    include("traversal.jl")

    include("queries/likelihood.jl")
    include("queries/likelihood_cpu.jl")
    include("queries/map_cpu.jl")
    include("queries/flow.jl")

    include("parameters/em.jl")

    include("io/io.jl")

    include("structures/hclts.jl")
    include("structures/rat.jl")
    
end
