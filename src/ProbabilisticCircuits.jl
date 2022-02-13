module ProbabilisticCircuits

    using DirectedAcyclicGraphs
    import DirectedAcyclicGraphs as DAGs # shorthand

    # reexport from DAGs
    export num_nodes, num_edges

    include("abstract_nodes.jl")
    
    include("input_distributions.jl")
    include("plain_nodes.jl")

    include("traversal.jl")

    include("structures/hclts.jl")

    include("queries/bits_circuit.jl")
    include("queries/likelihood.jl")
    include("queries/flow.jl")
    include("queries/em.jl")
    
end
