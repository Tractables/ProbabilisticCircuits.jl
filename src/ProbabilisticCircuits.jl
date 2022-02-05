# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

    using DirectedAcyclicGraphs
    import DirectedAcyclicGraphs as DAGs # shorthand

    # reexport from DAGs
    export num_nodes, num_edges

    include("abstract_prob_nodes.jl")
    include("input_distributions.jl")
    include("plain_prob_nodes.jl")

end
