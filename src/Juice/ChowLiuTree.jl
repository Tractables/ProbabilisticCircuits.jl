using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

include("../Utils/Utils.jl")
using .Utils

include("../Data/Data.jl")
using .Data

#####################
# Get mutual information
#####################

function marginal_distribution(vector, type_num, smoothing_factor)
    dis = Dict()
    len = length(vector)
    for x in vector
        dis[x] = get(dis, x, 0) + 1
    end
    for x in 0 : type_num - 1
        dis[x] = (get(dis, x, 0) + smoothing_factor * type_num) /
            (len + type_num * type_num * smoothing_factor)
    end
    return dis
end


function pairwise_distribution(vector1, vector2, type_num, smoothing_factor)
    @assert length(vector1) == length(vector2)
    dis = Dict()
    len = length(vector1)
    for i in 1 : len
        dis[(vector1[i], vector2[i])] = get(dis, (vector1[i], vector2[i]), 0) + 1
    end
    for x in 0 : type_num - 1, y in 0 : type_num - 1
        dis[(x, y)] = (get(dis, (x, y), 0) + smoothing_factor) /
            (len + type_num * type_num * smoothing_factor)
    end
    return dis
end


function mutual_information(vector1, vector2, type_num;
        base = ℯ, smoothing_factor = 0)
    prob_i = marginal_distribution(vector1, type_num, smoothing_factor)
    prob_j = marginal_distribution(vector2, type_num, smoothing_factor)
    prob_ij = pairwise_distribution(vector1, vector2, type_num, smoothing_factor)
    mi = 0.0
    for x in keys(prob_i), y in keys(prob_j)
        if !isapprox(0.0, prob_ij[(x, y)]; atol=eps(Float64), rtol=0)
            mi += prob_ij[(x, y)] * log(base, prob_ij[(x, y)] / (prob_i[x] * prob_j[y]))
        end
    end
    return mi
end


#####################
# Get CPTs of tree-structured BN
#####################

function get_cpt(data_matrix, type_num, parent_index, child_index; smoothing_factor = 0)
    child = data_matrix[:, child_index]
    prob_c = marginal_distribution(child, type_num, smoothing_factor)
    if isequal(parent_index, nothing)
        return prob_c
    end
    parent = data_matrix[:, parent_index]
    prob_p = marginal_distribution(parent, type_num, smoothing_factor)
    prob_pc = pairwise_distribution(parent, child, type_num, smoothing_factor)
    cpt = Dict()
    for p in keys(prob_p), c in keys(prob_c)
        if !isapprox(0.0, prob_p[p]; atol=eps(Float64), rtol=0)
            cpt[(c, p)] = prob_pc[(p, c)] / prob_p[p]
        end
    end
    return cpt
end


#####################
# Learn a Chow-Liu tree from data
#####################

function chow_liu_tree(data; smoothing_factor = 0)
    features_num = num_features(data)
    data_matrix = feature_matrix(data)
    type_num = maximum(data_matrix[:, 1]) + 1

    # Calculate mutual information matrix
    g = SimpleWeightedGraph(features_num)
    for i in 1:features_num, j in i+1:features_num
        v1, v2 = data_matrix[:, i], data_matrix[:, j]
        mi = mutual_information(v1, v2, type_num; smoothing_factor = smoothing_factor)
        add_edge!(g, i, j, - mi)
    end

    # Maximum spanning tree/ forest
    mst_edges = kruskal_mst(g)
    tree = SimpleWeightedGraph(features_num)
    for edge in mst_edges
        add_edge!(tree, src(edge), dst(edge), - weight(edge))
    end
    roots = [c[1] for c in connected_components(tree)]
    rooted_tree = SimpleDiGraph(features_num)
    for root in roots rooted_tree = union(rooted_tree, bfs_tree(tree, root)) end

    # Construct Chow-Liu tree with CPTs
    clt = MetaDiGraph(rooted_tree)
    set_prop!(clt, :description, "Chow-Liu Tree of Weighted Sample")
    ## add weights
    for edge in edges(clt)
        set_prop!(clt, edge, :weight, tree.weights[src(edge), dst(edge)])
    end
    ## set parent
    for root in roots set_prop!(clt, root, :parent, nothing) end
    for edge in edges(clt)
        set_prop!(clt, dst(edge), :parent, src(edge))
    end
    ## calculate cpts
    for v in vertices(clt)
        parent = get_prop(clt, v, :parent)
        cpt_matrix = get_cpt(data_matrix, type_num, parent, v; smoothing_factor = smoothing_factor)
        set_prop!(clt, v, :cpt, cpt_matrix)
    end
    return clt
end
