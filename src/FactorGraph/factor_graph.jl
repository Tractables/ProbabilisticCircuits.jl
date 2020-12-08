export FactorGraph, FactorNode, VarNode, fromUAI
abstract type FGNode end

mutable struct VarNode <: FGNode
    dim:: Int
end

mutable struct FactorNode <: FGNode
    neighbs:: Array{VarNode}
    factor::Dict{Array{Int}, Float64}
end


mutable struct FactorGraph
    facs::Array{FactorNode}
    vars::Array{VarNode}
end



function fromUAI(filename)
    # Load Lines 
    lines = collect(eachline(filename))
    # Lines don't actually mean anything, just take whitespace separated
    tokens = filter(x -> length(x) > 0, split(join(lines, " "), r"\s+"))
    numVars = parse(Int, tokens[2])

    # Create variable nodes
    cards = map(x -> parse(Int, x), tokens[3:numVars+2])
    varNodes = [VarNode(c) for c in cards]

    # Create factor nodes
    numFacs = parse(Int, tokens[3+numVars])
    curr_token = 4 + numVars
    # Read in the neighbour list for factors
    neighbs = Array{Array{Int}}(undef, numFacs)
    for fac_ind in 1:numFacs
        num_neighbs = parse(Int, tokens[curr_token])
        # We're 1-indexed in julia
        neighbs[fac_ind] = map(x -> parse(Int, x) + 1, tokens[curr_token+1:curr_token+num_neighbs])
        # Update where we're looking
        curr_token += num_neighbs + 1
    end
    # Read in factor description
    facNodes = Array{FactorNode}(undef, numFacs)
    for fac_ind in 1:numFacs
        # Create the indices for the map
        neighb_cards = map(x -> varNodes[x].dim, neighbs[fac_ind])
        num_fac_entries = reduce(*, neighb_cards)
        @show neighbs[fac_ind]
        @show neighb_cards
        labels = reshape(permutedims(collect(Base.product([1:n for n in (neighb_cards)]...)), reverse(1:length(neighb_cards))), num_fac_entries)
        # Convert tuples to arrays
        labels = map(x -> [x...], labels)
        values = map(x -> parse(Float64, x), tokens[curr_token+1:curr_token+num_fac_entries])
        factor = Dict(labels .=> values)
        facNodes[fac_ind] = FactorNode([varNodes[x] for x in neighbs[fac_ind]], factor)
        # Update where we're looking
        curr_token += num_fac_entries + 1
    end

    FactorGraph(facNodes, varNodes)
end

        
