export RegionGraph, random_region_graph, region_graph_2_pc
import Random: shuffle

"Root of region graph node hierarchy"
abstract type RegionGraph <: Tree end

######
## Partition
#####
mutable struct Partition 
    partition::AbstractVector{T} where T <: RegionGraph
    Partition(n::AbstractVector{T}) where T <: RegionGraph = begin
        new(n)
    end 
end

import Base: getindex, size, length
@inline Base.size(n::Partition) = size(n.partition)
@inline Base.length(n::Partition) = length(n.partition)
@inline Base.getindex(n::Partition, i::Int) = n.partition[i]


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

    # Each parition should include the same set of variables
    prev_scope = nothing
    for cur_partition in partitions
        cur_scope = variables(cur_partition[1])
        for i = 2:length(cur_partition)
            cur_scope = cur_scope ∪ variables(cur_partition[i])
        end
        
        if !isnothing(prev_scope)
            @assert prev_scope == cur_scope "All partitions should include the same set of variables."
        else
            prev_scope = cur_scope
        end
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


########################################
### Rat-SPNs in Juice PC data structure
########################################

"""
Makes sure the sum nodes does not have too many children. Makes balanced sums of sums to reduce children count.
"""
function balance_sum(children::Vector{ProbCircuit}, balance_childs_parents)::PlainSumNode
    if balance_childs_parents
        if length(children) <= 4
            PlainSumNode(children)
        else
            ls = 1:floor(Int32, length(children) / 2)
            rs = ceil(Int32, length(children) / 2):length(children)
            PlainSumNode([balance_sum(children[ls], balance_childs_parents), balance_sum(children[rs], balance_childs_parents)])
        end
    else
        PlainSumNode(children)
    end
end

"""
Makes sure input nodes don't have too many parents.
Makes a dummy sum node for each input per partition. Then nodes corresponding to the partition use
the dummy node as their children instead of the input node.
This way instead of num_nodes_root * num_nodes_leaf, we would have num_nodes_root parents nodes.
"""
function balanced_fully_factorized(vtrees::Vector{<:Vtree})::Vector{ProbCircuit}
    vars = variables(vtrees[1]) # assuming all vtrees same set of variables
    var_2_dummy_inputs = Dict(var => PlainSumNode([PlainProbLiteralNode(var), PlainProbLiteralNode(-var)]) for var in vars)
    
    balanced_recurse(nodes::Vector{<:Vtree})::Vector{ProbCircuit} = begin
        if isleaf(nodes[1])
            [PlainSumNode([var_2_dummy_inputs[variable(node)]]) for node in nodes]
        else
            lefts = balanced_recurse([node.left for node in nodes])
            rights = balanced_recurse([node.right for node in nodes])
            [PlainSumNode([PlainMulNode([left, right])]) for (left, right) in zip(lefts, rights)]
        end
    end

    balanced_recurse(vtrees)
end


"""
    region_graph_2_pc(node::RegionGraph; num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents)

- `num_nodes_root`: number of sum nodes in the root region
- `num_nodes_leaf`: number of sum nodes per leaf region
- `num_nodes_region`: number of in each region except root and leaves
"""
function region_graph_2_pc(node::RegionGraph; 
    num_nodes_root::Int = 4,  num_nodes_region::Int = 3, num_nodes_leaf::Int = 2, balance_childs_parents)::Vector{ProbCircuit}
    sum_nodes = Vector{ProbCircuit}()
    if isleaf(node)
        if false #balance_childs_parents # TODO fix, turning this on makes param learning faster but log likelihood improves much slower
            vtrees = [Vtree([PlainVtreeLeafNode(x) for x ∈ variables(node)], :balanced) for i=1:num_nodes_leaf]
            sum_nodes = balanced_fully_factorized(vtrees)
        else
            for i = 1:num_nodes_leaf
                vtree = Vtree([PlainVtreeLeafNode(x) for x ∈ variables(node)], :balanced)
                leaf_circuit = PlainProbCircuit(fully_factorized_circuit(ProbCircuit, vtree))
                push!(sum_nodes, leaf_circuit)
            end
        end
    else
        root_children = Vector{ProbCircuit}()
        # Foreach replication; usually only > 1 at root
        for partition in node.partitions
            partition_mul_nodes = Vector{ProbCircuit}()

            @assert length(partition) == 2 "Only supporting partitions of size 2 at the moment"
            lefts = region_graph_2_pc(partition[1]; num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents) 
            rights = region_graph_2_pc(partition[2]; num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents)
            @assert all([issum(l) for l in lefts])
            @assert all([issum(r) for r in rights])

            for l in lefts, r in rights
                mul_node = PlainMulNode([l, r])
                push!(partition_mul_nodes, mul_node)
            end

            dummy_sum_node = balance_sum(partition_mul_nodes, balance_childs_parents) 
            push!(root_children, dummy_sum_node)
        end

        # Repeat Sum nodes nodes based on where in Region Graph
        if isnothing(node.parent)
            # Root region
            for i = 1:num_nodes_root
                sum_node = balance_sum(root_children, balance_childs_parents)
                push!(sum_nodes, sum_node)
            end
        else
            # Inner region
            for i = 1:num_nodes_region
                sum_node = balance_sum(root_children, balance_childs_parents) 
                push!(sum_nodes, sum_node)
            end
        end
    end

    sum_nodes
end