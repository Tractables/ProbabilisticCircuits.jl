using BlossomV
using Metis
using SparseArrays
using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs
using .Juice.Utils

const δINT = 999999
const MIN_INT = 1
const MAX_INT = δINT + MIN_INT

#############
# Metis top down method
#############

struct MetisContext
    info::Matrix{Int64}
end

MetisContext(mi::Matrix{Float64}) = MetisContext(to_long_mi(mi, MIN_INT, MAX_INT))


#Add edge weights to Metis.jl

import Metis.idx_t
import Metis.partition
import Metis.ishermitian
import Metis.graph
import Metis.partition

struct WeightedGraph
    nvtxs::idx_t
    xadj::Vector{idx_t}
    adjncy::Vector{idx_t}
    adjwgt::Vector{idx_t} # edge weights
    WeightedGraph(nvtxs, xadj, adjncy, adjwgt) = new(nvtxs, xadj, adjncy, adjwgt)
end

function graph(G::SparseMatrixCSC; check_hermitian=true)
    if check_hermitian
        ishermitian(G) || throw(ArgumentError("matrix must be Hermitian"))
    end
    N = size(G, 1)
    xadj = Vector{idx_t}(undef, N+1)
    xadj[1] = 1
    adjncy = Vector{idx_t}(undef, nnz(G))
    adjncy_i = 0
    adjwgt = Vector{idx_t}(undef, nnz(G))
    @inbounds for j in 1:N
        n_rows = 0
        for k in G.colptr[j] : (G.colptr[j+1] - 1)
            i = G.rowval[k]
            if i != j # don't include diagonal elements
                n_rows += 1
                adjncy_i += 1
                adjncy[adjncy_i] = i
                adjwgt[adjncy_i] = G[i, j]
            end
        end
        xadj[j+1] = xadj[j] + n_rows
    end
    resize!(adjncy, adjncy_i)
    resize!(adjwgt, adjncy_i)
    return WeightedGraph(idx_t(N), xadj, adjncy, adjwgt)
end

function partition(G::WeightedGraph, nparts::Integer; alg = :KWAY)
    part = Vector{Metis.idx_t}(undef, G.nvtxs)
    edgecut = fill(idx_t(0), 1)
    if alg === :RECURSIVE
        Metis.METIS_PartGraphRecursive(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
                                 idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    elseif alg === :KWAY
        Metis.METIS_PartGraphKway(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
                            idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    else
        throw(ArgumentError("unknown algorithm $(repr(alg))"))
    end
    return part
end

partition(G, nparts; alg = :KWAY) = partition(graph(G), nparts, alg = alg)

"Metis top down method"
function metis_top_down(vars::Set{Var}, context::MetisContext)::Tuple{Set{Var}, Set{Var}}

    vertices = sort(collect(vars))
    sub_context = context.info[vertices, vertices]
    len = length(vertices)
    for i in 1 : len
        sub_context[i, i] = 0
    end
    g = convert(SparseMatrixCSC, sub_context)
    partition = Metis.partition(graph(g), 2, alg = :RECURSIVE)

    subsets = (Set{Var}(), Set{Var}())
    for (index, p) in enumerate(partition)
        push!(subsets[p], vertices[index])
    end

    return subsets
end

function metis_top_down_curry(context::MetisContext)
    f(vars) = metis_top_down(vars, context)
    return f
end


#############
# Blossom bottom up method
#############

# TODO change API to DisjointSet
mutable struct BlossomContext
    variable_sets::Vector{Vector{Var}}
    partition_id::Vector{Int64} # map vars to index in variable_sets
    info::Matrix
end

BlossomContext(vars::Set{Var}, mi::Matrix{Float64}) =
    BlossomContext( [[v] for v in sort(collect(vars))],
                    collect(1 : length(vars)),
                    round.(Int64, 1000001 .+ to_long_mi(mi, -1, -1000000)))
                    #mi)

"Blossom bottom up method, vars are not used"
function blossom_bottom_up!(vars::Set{Var}, context::BlossomContext)::Set{Tuple{Var, Var}}

    "even number of nodes, use blossomv alg"
    function blossom_bottom_up_even!(vars::Set{Var}, context::BlossomContext; update = true)::Tuple{Set{Tuple{Var, Var}}, Int64}
        "1. calculate pMI"
        pMI = set_mutual_information(context.info, context.variable_sets)
        #pMI = 1000001 .+ to_long_mi(pMI, -1, -1000000)
        pMI = round.(Int64, pMI)

        "2. solve by blossomv alg"
        len = length(context.variable_sets)
        m = Matching(len)
        for i in 1 : len, j in i + 1 : len
            add_edge(m, i - 1, j - 1, pMI[i, j]) # blossomv index start from 0
        end

        solve(m)
        all_matches = Set{Tuple{Var, Var}}()
        for v in 1 : len
            push!(all_matches, order_asc(v, get_match(m, v - 1) + 1))
        end

        "3. calculate scores, map index to var"
        all_matches = Vector(collect(all_matches))
        score = 0

        for i in 1 : length(all_matches)
            (x, y) = all_matches[i]
            score += pMI[x, y]
            all_matches[i] = (context.variable_sets[x][1], context.variable_sets[y][1])
        end
        all_matches = Set(all_matches)

        "4. update context when called by outer layer"
        if update
            updata_context(all_matches, context)
        end

        return (all_matches, score)
    end

    "odd number of nodes, try every 2 combinations"
    function blossom_bottom_up_odd!(vars::Set{Var}, context::BlossomContext)::Tuple{Set{Tuple{Var, Var}}, Int64}

        "1. try all len - 1 conditions, find best score(minimun cost)"
        (best_matches, best_score) = (Set{Tuple{Var, Var}}(), typemax(Int64))

        for index in 1 : length(context.variable_sets)
            sub_context = deepcopy(context)
            set = copy(sub_context.variable_sets[index])
            deleteat!(sub_context.variable_sets, index)

            (matches, score) = blossom_bottom_up_even!(vars, sub_context; update = false)
            if score < best_score
                (best_matches, best_score) = (matches, score)
            end

            insert!(sub_context.variable_sets, index, set)
        end

        "2. update information"
        updata_context(best_matches, context)
        return (best_matches, best_score)

    end

    function updata_context(matches::Set{Tuple{Var, Var}}, context::BlossomContext)
        for (x, y) in matches
            y_partition = copy(context.variable_sets[context.partition_id[y]])
            context.variable_sets[context.partition_id[y]] = Vector()
            foreach(value -> push!(context.variable_sets[context.partition_id[x]], value), y_partition)
        end

        context.variable_sets = [x for x in context.variable_sets if x != []]
        for index in 1 : length(context.variable_sets)
            for y in context.variable_sets[index]
                context.partition_id[y] = index
            end
        end
    end

    if length(vars) % 2 == 0
        (matches, score) = blossom_bottom_up_even!(vars, context)
    else
        (matches, score) = blossom_bottom_up_odd!(vars, context)
    end

    for (left, right) in matches
        pop!(vars, right)
    end

    return matches
end

function blossom_bottom_up_curry(context::BlossomContext)
    f(vars) = blossom_bottom_up(vars, context)
    return f
end


#############
# Test method
#############


"Test top down method, split nodes by ascending order, balanced"
function test_top_down(vars::Set{Var})::Tuple{Set{Var}, Set{Var}}
    sorted_vars = sort(collect(vars))
    len = length(sorted_vars)
    len1 = Int64(len % 2 == 0 ? len // 2 : (len - 1) // 2)
    return (Set(sorted_vars[1 : len1]), Set(sorted_vars[len1 + 1 : end]))
end

"Test bottom up method, split nodes by ascending order, balanced"
function test_bottom_up!(vars::Set{Var})::Set{Tuple{Var, Var}}
    sorted_vars = sort(collect(vars))
    len = length(sorted_vars)
    len1 = Int64(len % 2 == 0 ? len // 2 : (len - 1) // 2)
    matches = Set{Tuple{Var, Var}}()
    for i in 1 : len1
        push!(matches, (sorted_vars[2 * i - 1], sorted_vars[2 * i]))
        pop!(vars, sorted_vars[2 * i])
    end
    return matches
end

#############
# Learn Vtree from CLT
#############

function learn_vtree_from_clt(clt::CLT, strategy::String)::Vtree
    roots = filter(x->x==0,parent_vector(clt))
    root = construct_children(Var.(roots), clt, strategy)

    return order_nodes_leaves_before_parents(root)
end

function construct_node(v::Var, clt::CLT, strategy::String)::VtreeNode
    children = Var.(outneighbors(clt, v))
    if isempty(children) # leaf node
        return VtreeLeafNode(v)
    else
        right = construct_children(children, clt, strategy)
        return add_parent(v, right)
    end
end

function construct_children(children::Vector{Var}, clt::CLT, strategy::String)::VtreeNode
    sorted_vars = sort(collect(children))
    children_nodes = Vector{VtreeNode}()
    foreach(x -> push!(children_nodes, construct_node(x, clt, strategy)), sorted_vars)

    if strategy == "linear"
        construct_children_linear(children_nodes, clt)
    elseif strategy == "balanced"
        construct_children_balanced(children_nodes, clt)
    else
        throw("Unknown type of strategy")
    end
end

function construct_children_linear(children_nodes::Vector{VtreeNode}, clt::CLT)::VtreeNode
    children_nodes = Iterators.Stateful(reverse(children_nodes))

    right = popfirst!(children_nodes)
    for left in children_nodes
        right = VtreeInnerNode(left, right)
    end
    return right
end

function construct_children_balanced(children_nodes::Vector{VtreeNode}, clt::CLT)::VtreeNode
    if length(children_nodes) == 1
        return children_nodes[1]
    elseif length(children_nodes) == 2
        return VtreeInnerNode(children_nodes[1], children_nodes[2])
    else
        len = trunc(Int64, length(children_nodes) / 2)
        left = construct_children_balanced(children_nodes[1 : len], clt)
        right = construct_children_balanced(children_nodes[len + 1 : end], clt)
        return VtreeInnerNode(left, right)
    end
end

function add_parent(parent::Var, children::VtreeNode)
    return VtreeInnerNode(VtreeLeafNode(parent), children)
end
