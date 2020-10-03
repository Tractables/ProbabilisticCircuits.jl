using BlossomV
using Metis
using SparseArrays
using LightGraphs: add_edge!
using SimpleWeightedGraphs
using MetaGraphs
using ..Utils

export learn_vtree

#############
# Metis top down method
#############

# Add edge weights to Metis.jl
using Metis: idx_t, ishermitian

struct WeightedGraph
    nvtxs::idx_t
    xadj::Vector{idx_t}
    adjncy::Vector{idx_t}
    adjwgt::Vector{idx_t} # edge weights
    WeightedGraph(nvtxs, xadj, adjncy, adjwgt) = new(nvtxs, xadj, adjncy, adjwgt)
end

function my_graph(G::SparseMatrixCSC; check_hermitian=true)
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

function my_partition(G::WeightedGraph, nparts::Integer; alg = :KWAY)
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

my_partition(G, nparts; alg = :KWAY) = my_partition(my_graph(G), nparts, alg = alg)

"Metis top down method"
function metis_top_down(data::DataFrame;α)
    δINT = 999999
    MIN_INT = 1
    MAX_INT = δINT + MIN_INT

    weight=ones(Float64, num_examples(data))
    (_, mi) = mutual_information(data, weight; α=α)
    vars = Var.(collect(1:num_features(data)))
    info = to_long_mi(mi, MIN_INT, MAX_INT)

    function f(leafs::Vector{PlainVtreeLeafNode})::Tuple{Vector{PlainVtreeLeafNode}, Vector{PlainVtreeLeafNode}}
        var2leaf = Dict([(variable(x),x) for x in leafs])
        vertices = sort(variable.(leafs))
        sub_context = info[vertices, vertices]
        len = length(vertices)
        for i in 1 : len
            sub_context[i, i] = 0
        end
        g = convert(SparseMatrixCSC, sub_context)
        partition = my_partition(my_graph(g), 2, alg = :RECURSIVE)
    
        subsets = (Vector{PlainVtreeLeafNode}(), Vector{PlainVtreeLeafNode}())
        for (index, p) in enumerate(partition)
            push!(subsets[p], var2leaf[vertices[index]])
        end
    
        return subsets
    end
    return f
end

#############
# Blossom bottom up method
#############

"Blossom bottom up method, vars are not used"
function blossom_bottom_up(data::DataFrame;α)
    weight = ones(Float64, num_examples(data))
    (_, mi) = mutual_information(data, weight; α)
    vars = Var.(collect(1:num_features(data)))
    info = round.(Int64, 1000001 .+ to_long_mi(mi, -1, -1000000))

    function f(leaf::Vector{<:Vtree})
        variable_sets = collect.(variables.(leaf))
        
        # even number of nodes, use blossomv alg
        function blossom_bottom_up_even!(variable_sets)::Tuple{Vector{Tuple{Var, Var}}, Int64}
            # 1. calculate pMI
            pMI = set_mutual_information(info, variable_sets)
            pMI = round.(Int64, pMI)

            # 2. solve by blossomv alg
            len = length(variable_sets)
            m = Matching(len)
            for i in 1 : len, j in i + 1 : len
                add_edge(m, i - 1, j - 1, pMI[i, j]) # blossomv index start from 0
            end

            solve(m)
            all_matches = Set{Tuple{Var, Var}}()
            for v in 1 : len
                push!(all_matches, order_asc(v, get_match(m, v - 1) + 1))
            end

            # 3. calculate scores, map index to var
            all_matches = Vector(collect(all_matches))
            score = 0

            for i in 1 : length(all_matches)
                (x, y) = all_matches[i]
                score += pMI[x, y]
            end

            return (all_matches, score)
        end

        # odd number of nodes, try every 2 combinations
        function blossom_bottom_up_odd!(variable_sets)
            # try all len - 1 conditions, find best score(minimun cost)
            (best_matches, best_score) = (nothing, typemax(Int64))
            len = length(variable_sets)
            for index in 1 : len
                indices = [collect(1:index-1);collect(index+1:len)]
                (matches, score) = blossom_bottom_up_even!(variable_sets[indices])
                if score < best_score
                    (best_matches, best_score) = ([[(indices[l], indices[r]) for (l,r) in matches];[index]], score)
                end
            end
            return (best_matches, best_score)
        end

        if length(variable_sets) % 2 == 0
            (matches, score) = blossom_bottom_up_even!(variable_sets)
        else
            (matches, score) = blossom_bottom_up_odd!(variable_sets)
        end
    
        pairs = []
        for x in matches
            if x isa Tuple
                push!(pairs, (leaf[x[1]], leaf[x[2]]))
            else
                push!(pairs, leaf[x])
            end
        end
        return pairs
    end
    return f
end

function learn_vtree(data::DataFrame; α=0.0, alg=:bottomup)
    if alg==:topdown
        PlainVtree(num_features(data), :topdown; f=metis_top_down(data;α))
    elseif alg==:bottomup
        PlainVtree(num_features(data), :bottomup; f=blossom_bottom_up(data;α))
    elseif alg==:clt
        clt = learn_chow_liu_tree(data)
        learn_vtree_from_clt(clt, vtree_mode="balanced")
    else
        error("Vtree learner $(alg) not supported.")
    end
end