using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

#####################
# Learn a Chow-Liu tree from weighted data
#####################

const CLT = Union{MetaDiGraph, SimpleDiGraph}

"""
learn a Chow-Liu tree from training set `train_x`, with Laplace smoothing factor `α`,
for simplification, if `parametered=false`, CPTs are not cached in vertices,
to get parameters, run `learn_prob_circuit` as wrapper;
if `parametered=true`, cache CPTs in vertices.
"""
function learn_chow_liu_tree(train_x::XData; α = 0.0001, parametered=true)
    learn_chow_liu_tree(WXData(train_x);α = α, parametered = parametered)
end

function learn_chow_liu_tree(train_x::WXData; α = 0.0001, parametered = true)
    features_num = num_features(train_x)

    # calculate mutual information
    (dis_cache, MI) = mutual_information(train_x; α = α)

    # maximum spanning tree/ forest
    g = SimpleWeightedGraph(CompleteGraph(features_num))
    mst_edges = kruskal_mst(g,-MI)
    tree = SimpleGraph(features_num)
    map(mst_edges) do edge
        add_edge!(tree, src(edge), dst(edge))
    end

    # Build rooted tree / forest
    roots = [c[1] for c in connected_components(tree)]
    clt = SimpleDiGraph(features_num)
    for root in roots clt = union(clt, bfs_tree(tree, root)) end

    # if parametered, cache CPTs in vertices
    if parametered
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
    end

    return clt
end

"Get parent vector of a tree"
function parent_vector(tree::CLT)::Vector{Int64}
    v = zeros(Int64, nv(tree)) # parent of roots is 0
    foreach(e->v[dst(e)] = src(e), edges(tree))
    return v
end

#####################
# Methods for test
#####################
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

"Parse a clt from given file"
function parse_clt(filename::String)::MetaDiGraph
    f = open(filename)
    n = parse(Int32,readline(f))
    n_root = parse(Int32,readline(f))
    clt = MetaDiGraph(n)
    for i in 1 : n_root
        root, prob = split(readline(f), " ")
        root, prob = parse(Int32, root), parse(Float64, prob)
        set_prop!(clt, root, :parent, 0)
        set_prop!(clt, root, :cpt, Dict(1=>prob,0=>1-prob))
    end

    for i = 1 : n - n_root
        dst, src, prob1, prob0 = split(readline(f), " ")
        dst, src, prob1, prob0 = parse(Int32, dst), parse(Int32, src), parse(Float64, prob1), parse(Float64, prob0)
        add_edge!(clt, src,dst)
        set_prop!(clt, dst, :parent, src)
        set_prop!(clt, dst, :cpt, Dict((1,1)=>prob1, (0,1)=>1-prob1, (1,0)=>prob0, (0,0)=>1-prob0))
    end
    return clt
end
