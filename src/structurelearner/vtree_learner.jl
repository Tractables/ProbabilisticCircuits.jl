using Metis
using SparseArrays
using SimpleWeightedGraphs
using MetaGraphs: add_edge!
using ..Utils

export learn_vtree

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

function my_partition(G::WeightedGraph, nparts::Integer)
    part = Vector{Metis.idx_t}(undef, G.nvtxs)
    edgecut = fill(idx_t(0), 1)
    # if alg === :RECURSIVE
        Metis.METIS_PartGraphRecursive(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
                                 idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    # elseif alg === :KWAY
    #     Metis.METIS_PartGraphKway(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
    #                         idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    # else
    #     throw(ArgumentError("unknown algorithm $(repr(alg))"))
    # end
    return part
end

my_partition(G, nparts) = my_partition(my_graph(G), nparts)

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
        partition = my_partition(my_graph(g), 2)
    
        subsets = (Vector{PlainVtreeLeafNode}(), Vector{PlainVtreeLeafNode}())
        for (index, p) in enumerate(partition)
            push!(subsets[p], var2leaf[vertices[index]])
        end
    
        return subsets
    end
    return f
end
