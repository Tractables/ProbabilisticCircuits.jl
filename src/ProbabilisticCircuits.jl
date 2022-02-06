module ProbabilisticCircuits

    using DirectedAcyclicGraphs
    import DirectedAcyclicGraphs as DAGs # shorthand

    # reexport from DAGs
    export num_nodes, num_edges

    include("abstract_nodes.jl")
    
    include("input_distributions.jl")
    include("plain_nodes.jl")

    include("traversal.jl")

end
