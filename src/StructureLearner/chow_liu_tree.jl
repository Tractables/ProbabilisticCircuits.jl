export CLT, learn_chow_liu_tree, parent_vector
using LightGraphs: SimpleGraph, SimpleDiGraph, complete_graph, add_edge!, kruskal_mst, 
    bfs_tree, center, connected_components, induced_subgraph, nv, ne, edges, vertices, src, dst
using SimpleWeightedGraphs: SimpleWeightedGraph
using MetaGraphs: MetaDiGraph, set_prop!, props

#####################
# Learn a Chow-Liu tree from (weighted) data
#####################

"""
Chow-Liu Tree
"""
const CLT = MetaDiGraph

"""
learn a Chow-Liu tree from training set `train_x`, with Laplace smoothing factor `α`, specifying the tree root by `clt_root`
return a `CLT`
"""
function learn_chow_liu_tree(train_x; α = 1.0, clt_root="graph_center",
        weight=ones(Float64, num_examples(train_x)))::CLT
    features_num = num_features(train_x)

    # calculate mutual information
    (dis_cache, MI) = mutual_information(train_x, weight; α = α)

    # maximum spanning tree/ forest
    g = SimpleWeightedGraph(complete_graph(features_num))
    mst_edges = kruskal_mst(g,- MI)
    tree = SimpleGraph(features_num)
    map(mst_edges) do edge
        add_edge!(tree, src(edge), dst(edge))
    end

    # Build rooted tree / forest
    if clt_root == "graph_center"
        clt = SimpleDiGraph(features_num)
        if nv(tree) == ne(tree) + 1
            clt = bfs_tree(tree, center(tree)[1])
        else
            for c in filter(c -> (length(c) > 1), connected_components(tree))
                sg, vmap = induced_subgraph(tree, c)
                sub_root = vmap[center(sg)[1]]
                clt = union(clt, bfs_tree(tree, sub_root))
            end
        end
    elseif clt_root == "rand"
        roots = [rand(c) for c in connected_components(tree)]
        clt = SimpleDiGraph(features_num)
        for root in roots clt = union(clt, bfs_tree(tree, root)) end
    else
        error("Cannot learn CLT with root $clt_root")
    end
    
    clt = MetaDiGraph(clt)
    parent = parent_vector(clt)
    for (c, p) in enumerate(parent)
        set_prop!(clt, c, :parent, p)
    end

    for v in vertices(clt)
        p = parent[v]
        cpt_matrix = get_cpt(p, v, dis_cache)
        set_prop!(clt, v, :cpt, cpt_matrix)
    end

    return clt
end

"""
Calculate CPT of `child` conditioned on `parent` from `dis_cache`
"""
function get_cpt(parent, child, dis_cache)
    if parent == 0
        p = dis_cache.marginal[child, :]
        return Dict(0=>p[1], 1=>p[2])
    else
        p = dis_cache.pairwise[child, parent, :] ./ [dis_cache.marginal[parent, :]; dis_cache.marginal[parent, :]]
        @. p[isnan(p)] = 0; @. p[p==Inf] = 0; @. p[p == -Inf] = 0
        return Dict((0,0)=>p[1], (1,0)=>p[3], (0,1)=>p[2], (1,1)=>p[4]) #p(child|parent)
    end
end


"Get parent vector of a tree"
function parent_vector(tree::CLT)::Vector{Int64}
    v = zeros(Int64, nv(tree)) # parent of roots is 0
    foreach(e->v[dst(e)] = src(e), edges(tree))
    return v
end

import LogicCircuits: print_tree
"Print edges and vertices of a ChowLiu tree"
function print_tree(clt::CLT)
    for e in edges(clt) print(e); print(" ");end
    if clt isa SimpleDiGraph
        for v in vertices(clt) print(v); print(" "); end
    end
    if clt isa MetaDiGraph
        for v in vertices(clt) print(v); print(" "); println(props(clt, v)) end
    end
end
