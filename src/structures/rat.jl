export RegionGraph, 
       random_region_graph, 
       region_graph_2_pc, 
       RAT_InputFunc,
       RAT

import Random: shuffle
import DirectedAcyclicGraphs: Tree

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

RegionGraph(v::AbstractVector) = RegionGraphLeafNode(BitSet(v))
RegionGraph(partitions::AbstractVector{Partition}) = RegionGraphInnerNode(partitions)

#####################
# Traits
#####################

@inline DirectedAcyclicGraphs.NodeType(::Type{<:RegionGraphInnerNode}) = Inner()
@inline DirectedAcyclicGraphs.NodeType(::Type{<:RegionGraphLeafNode}) = Leaf()

###################
## Methods
#################


import DirectedAcyclicGraphs: children

@inline children(n::RegionGraphInnerNode) = n.partitions
@inline variables(n::RegionGraphInnerNode) = n.variables
@inline variables(n::RegionGraphLeafNode) = n.variables

##################################################################

"""
    random_region_graph(X::AbstractVector, depth::Int = 5, replicas::Int = 2, num_splits::Int = 2)

- `X`: Vector of all variables to include; for the root region
- `depth`: how many layers to do splits
- `replicas`: number of replicas or paritions (replicas only used for the root region; for other regions only 1 parition (inner nodes), or 0 parition for leaves)
- `num_splits`: number of splits for each parition; split variables into random equaly sized regions
"""
function random_region_graph(X::AbstractVector;
    depth::Int = 5, replicas::Int = 2, num_splits::Int = 2)::RegionGraph

    if length(X) < 2 || depth == 0
        # Cannot/should not split anymore
        return RegionGraph(X)
    end

    partitions = Vector{Partition}()
    for repeat = 1 : replicas
        cur_rg = split_rg(X, depth; num_splits=num_splits)
        push!(partitions, cur_rg)
    end

    # Validation: Each Partition should include the same set of variables
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

function split_rg(variables::AbstractVector, depth::Int; num_splits::Int = 2)::Partition
    partition(x, n) = begin
        ## TODO; might not work if num_splits > 2
        sz = ceil(Int, (length(x) / n))
        [ x[i:min(i+sz-1, length(x))] for i = 1:sz:length(x) ]
    end 

    shuffle_variables = shuffle(variables)
    splits = partition(shuffle_variables, num_splits)
    cur_partition_regions = Vector{RegionGraph}()
    for split in splits
        # only 1 replicas for non-root
        rg_node = random_region_graph(split; depth=depth-1, replicas=1, num_splits)
        push!(cur_partition_regions, rg_node)
    end
    Partition(cur_partition_regions)
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
function balanced_fully_factorized_leaves(variables::AbstractVector; input_func::Function, num_nodes_leaf)::Vector{ProbCircuit}

    var_2_dummy_inputs(var) = begin         
        input_func(var)
    end
    
    balanced_recurse(variables::AbstractVector)::Vector{ProbCircuit} = begin
        # Leaf
        if length(variables) == 1
            [PlainSumNode([var_2_dummy_inputs(variables[1])]) for node=1:num_nodes_leaf]
        else
            mid = length(variables) ÷ 2
            lefts = balanced_recurse(variables[1:mid])
            rights = balanced_recurse(variables[mid+1:end])
            [PlainSumNode([PlainMulNode([left, right])]) for (left, right) in zip(lefts, rights)]
        end
    end
    
    balanced_recurse(variables)
end


"""
    region_graph_2_pc(node::RegionGraph; num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents)

- `num_nodes_root`: number of sum nodes in the root region
- `num_nodes_leaf`: number of sum nodes per leaf region
- `num_nodes_region`: number of in each region except root and leaves
"""
function region_graph_2_pc(node::RegionGraph; input_func::Function,
    num_nodes_root::Int = 4,  num_nodes_region::Int = 3, num_nodes_leaf::Int = 2, balance_childs_parents=true)::Vector{ProbCircuit}
    sum_nodes = Vector{ProbCircuit}()
    if isleaf(node)
        vars = Vector(collect(variables(node)))
        sum_nodes = balanced_fully_factorized_leaves(vars; num_nodes_leaf, input_func)
    else
        root_children = Vector{ProbCircuit}()
        # Foreach replication; usually only > 1 at root
        for partition in node.partitions
            partition_mul_nodes = Vector{ProbCircuit}()

            @assert length(partition) == 2 "Only supporting partitions of size 2 at the moment"
            lefts = region_graph_2_pc(partition[1]; input_func, num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents) 
            rights = region_graph_2_pc(partition[2]; input_func, num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents)
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


"""
    RAT(num_features; input_func::Function = RAT_InputFunc(Literal), num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas, num_nodes_root = 1, balance_childs_parents = true)

Generate a RAT-SPN structure. First, it generates a random region graph with `depth`, and `replicas`. 
Then uses the random region graph to generate a ProbCircuit conforming to that region graph.

- `num_features`: Number of features in the dataset, assuming x_1...x_n
- `input_func`: Function to generate a new input node for variable when calling `input_func(var)`.

The list of hyperparamters are:
- `rg_depth`: how many layers to do splits in the region graph
- `rg_replicas`: number of replicas or paritions (replicas only used for the root region; for other regions only 1 parition (inner nodes), or 0 parition for leaves)
- `num_nodes_root`: number of sum nodes in the root region
- `num_nodes_leaf`: number of sum nodes per leaf region
- `num_nodes_region`: number of in each region except root and leaves
- `num_splits`: number of splits for each parition; split variables into random equaly sized regions
"""
function RAT(num_features; input_func::Function = RAT_InputFunc(Literal), num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas, num_nodes_root = 1, balance_childs_parents = false)
    region_graph = random_region_graph([Var(i) for i=1: num_features]; depth=rg_depth, replicas=rg_replicas);
    circuit = region_graph_2_pc(region_graph; input_func, num_nodes_root, num_nodes_region, num_nodes_leaf, balance_childs_parents)[1];
    init_parameters(circuit; perturbation = 0.4)
    return circuit
end


"""
Default `input_func` for different types. This function returns another function `input_func`.
Then `input_func(var)` should generate a new input function with the desired distribution.
"""
function RAT_InputFunc(input_type::Type, args...)
    if input_type == Literal
        function lit_func(var)
            PlainSumNode([
                    InputNode(var, Literal(true)), 
                    InputNode(var, Literal(false))])
        end
        return lit_func
    elseif input_type == Categorical
        function cat_func(var)
            InputNode(var, Categorical(args...))
        end
        return cat_func
    elseif input_type == Binomial
        function bin_func(var)
            InputNode(var, Binomial(args...))
        end
        return bin_func
    else
        @assert false "No default `input_func` for Input Type $(input_type)."
    end
end