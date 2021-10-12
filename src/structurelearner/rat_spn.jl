

export RegionGraph, random_region_graph
import Random: shuffle

"Root of region graph node hierarchy"
abstract type RegionGraph <: Tree end

import Base: getindex, size, length
@inline Base.size(n::Partition) = size(n.partition)
@inline Base.length(n::Partition) = length(n.partition)
@inline Base.getindex(n::Partition, i::Int) = n.partition[i]


######
## Partition
#####
mutable struct Partition 
    partition::AbstractVector{T} where T <: RegionGraph
    Partition(n::AbstractVector{T}) where T <: RegionGraph = begin
        new(n)
    end 
end

mutable struct RegionGraphInnerNode <: RegionGraph
    partitions::AbstractVector{Partition}
    variables::BitSet
    parent::Union{Nothing, RegionGraphInnerNode}

    RegionGraphInnerNode(partitions::AbstractVector{Partition}) = begin
        for partition in partitions
            for i = 1:length(partition), j = i+1 : length(partition)
                @assert isdisjoint(variables(partition[i]), variables(partition[j]))
            end
        end
        scope = variables(partitions[1][1])
        for ind = 2:size(partitions[1])[1]
            scope = scope ∪ variables(partitions[1][ind])
        end

        #TODO add more assertions, specifically when more than one paritition
        # they should be on the same set of variables

        this = new(partitions, scope, nothing)
        for partition in partitions
            for i = 1:size(partition)[1]
                @assert isnothing(partition[i].parent)
                partition[i].parent = this
            end
        end
        this
    end
end


mutable struct RegionGraphLeafNode <: RegionGraph
    variables::BitSet
    parent::Union{Nothing, RegionGraphInnerNode}
    RegionGraphLeafNode(v) = new(v, nothing)
end


#####################
# Constructors
#####################

RegionGraph(v::AbstractVector{Var}) = RegionGraphLeafNode(BitSet(v))
RegionGraph(partitions::AbstractVector{Partition}) = RegionGraphInnerNode(partitions)

#####################
# Traits
#####################

import LogicCircuits.NodeType
@inline LogicCircuits.NodeType(::RegionGraphInnerNode) = Inner()
@inline LogicCircuits.NodeType(::RegionGraphLeafNode) = Leaf()

###################
## Methods
#################

import LogicCircuits.Utils: children, variables

@inline children(n::RegionGraphInnerNode) = n.partitions
@inline variables(n::RegionGraphInnerNode) = n.variables
@inline variables(n::RegionGraphLeafNode) = n.variables

##################################################################

"""
    random_region_graph(X::AbstractVector{Var}, depth::Int = 5, replicas::Int = 2, num_splits::Int = 2)

- `X`: Vector of all variables to include; for the root region
- `depth`: how many layers to do splits
- `replicas`: number of replicas or paritions (replicas only used for the root region; for other regions only 1 parition (inner nodes), or 0 parition for leaves)
- `num_splits`: number of splits for each parition; split variables into random equaly sized regions
"""
function random_region_graph(X::AbstractVector{Var};
    depth::Int = 5, replicas::Int = 2, num_splits::Int = 2)::RegionGraph
    partitions = Vector{Partition}()
    for repeat = 1 : replicas
        cur_rg = split_rg(X, depth; num_splits=num_splits, return_partition_only = true)
        push!(partitions, cur_rg)
    end
    RegionGraphInnerNode(partitions)
end

function split_rg(variables::AbstractVector{Var}, depth::Int; num_splits::Int = 2, return_partition_only = false)
    if length(variables) < 2 || depth == 0
        # Cannot/shoud not split anymore
        return RegionGraph(variables)
    end

    shuffle_variables = shuffle(variables)
    
    partition(x, n) = begin
        ## TODO; might not work if num_splits > 2
        sz = ceil(Int, (length(x) / n))
        [ x[i:min(i+sz-1, length(x))] for i = 1:sz:length(x) ]
    end 
    splits = partition(shuffle_variables, num_splits)

    answer = Vector{RegionGraph}()
    for split in splits
        push!(answer, split_rg(split, depth-1; num_splits = num_splits))
    end

    if return_partition_only
        Partition(answer)
    else
        RegionGraphInnerNode([Partition(answer)])
    end
end