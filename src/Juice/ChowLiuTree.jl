using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs
using DataStructures

#####################
# Learn a Chow-Liu tree from weighted data
#####################

"learn a Chow-Liu tree from data matrix, with Laplace smoothing factor α"
function learn_chow_liu_tree(data::XData; α = 0.0001, parametered = true)
    learn_chow_liu_tree(WXData(data);α = α, parametered = parametered)
end

function learn_chow_liu_tree(data::WXData; α = 0.0001, parametered = true)
    weight_vector = Data.weights(data)
    data_matrix = feature_matrix(data)
    features_num = num_features(data)
    type_num = 2 #binary dataset
    dis_cache = DisCache()

    # Calculate all distrubutions and mutual information
    (dis_cache, MI) = mutual_information(data; type_num = type_num, α = α)

    # Maximum spanning tree/ forest
    g = SimpleWeightedGraph(features_num)
    for i in 1 : features_num, j in i + 1 : features_num
        add_edge!(g, i, j, - MI[i, j])
    end

    mst_edges = kruskal_mst(g)
    tree = SimpleWeightedGraph(features_num)
    for edge in mst_edges
        add_edge!(tree, src(edge), dst(edge), - weight(edge))
    end

    # Build rooted tree / forest
    roots = [c[1] for c in connected_components(tree)]
    rooted_tree = SimpleDiGraph(features_num)
    for root in roots rooted_tree = union(rooted_tree, bfs_tree(tree, root)) end

    # Construct Chow-Liu tree with CPTs
    clt = MetaDiGraph(rooted_tree)
    set_prop!(clt, :description, "Chow-Liu Tree of Weighted Sample")
    set_prop!(clt, :parametered, false)

    ## set parent
    for root in roots set_prop!(clt, root, :parent, 0) end
    for edge in edges(clt)
        set_prop!(clt, dst(edge), :parent, src(edge))
    end

    # By default, does not cache CPT in CLT to save computation, run *estimate_parameters* on ProbCircuits instead
    if parametered
        set_prop!(clt, :parametered, true)

        ## add weights
        for edge in edges(clt)
            set_prop!(clt, edge, :weight, tree.weights[src(edge), dst(edge)])
        end

        ## calculate cpts
        for v in vertices(clt)
            parent = get_prop(clt, v, :parent)
            cpt_matrix = get_cpt(parent, v, dis_cache)
            set_prop!(clt, v, :cpt, cpt_matrix)
        end

    end
        return clt

end

#####################
# Other methods about CLT
#####################

"Print edges and vertices of a ChowLiu tree"
function print_tree(clt)
    for e in edges(clt) print(e); print(" "); println(props(clt, e)) end
    for v in vertices(clt) print(v); print(" "); println(props(clt, v)) end
end

"Calculate complete disjoint probability for one query"
function clt_get_log_likelihood(clt, query)
    @assert(length(query) == nv(clt))
    @assert get_prop(clt, :parametered)
    probability = 0.0

    for v in vertices(clt)
        parent = get_prop(clt, v, :parent)
        cpt = get_prop(clt, v, :cpt)

        if parent == 0
            probability += log(cpt[query[v]])
        else
            probability += log(cpt[(query[v], query[parent])])
        end
    end

    return probability
end

"Calculate log probability for every sample"
function clt_log_likelihood_per_instance(clt, data)
    @assert get_prop(clt, :parametered)
    data_matrix = feature_matrix(data)
    num_sample = size(data_matrix)[1]
    features_num = num_features(data)
    result = zeros(num_sample)

    roots = [v for v in vertices(clt) if get_prop(clt, v, :parent) == 0]
    others = [v for v in vertices(clt) if get_prop(clt, v, :parent) != 0]

    for v in roots
        cpt = get_prop(clt, v, :cpt)
        result += log.([cpt[data_matrix[i, v]] for i in 1 : num_sample])
    end

    for v in others
        cpt = get_prop(clt, v, :cpt)
        parent = get_prop(clt, v, :parent)
        result += log.([cpt[(data_matrix[i, v], data_matrix[i, parent])] for i in 1 : num_sample])
    end

    return result
end

"Parse a chow-liu tree"
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

"Stable hierarchy order by bfs, parent before children, stable means children share the same parents ordered by alphabetical order"
function stable_hierarchy_order(clt::MetaDiGraph)
    roots = [v for v in vertices(clt) if get_prop(clt, v, :parent) == 0]
    sort!(roots)

    order = Vector{Int64}()
    queue = Queue{Int64}()
    foreach(x -> enqueue!(queue, x), roots)

    while !isempty(queue)
        cur = dequeue!(queue)
        push!(order, cur)

        children = sort(neighbors(clt, cur))
        foreach(x -> enqueue!(queue, x), children)
    end

    return order
end

"get parent vector of a tree"
function parent_vector(tree::MetaDiGraph)::Vector{Int64}
    v = zeros(Int64, nv(tree))
    for e in edges(tree)
        v[dst(e)] = src(e)
    end
    return v
end
