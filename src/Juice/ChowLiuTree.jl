using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

#TODO:(mhdang) accelarate speed by cache pairwise and marginal information
#####################
# Get mutual information
#####################
"calculate marginal distribution of weighted array, for consistency with pairwise distribution, type_num multiply twice"
function marginal_distribution(vector::AbstractArray, weight::Array,
        type_num::Int,smoothing_factor::Real)::Dict
    @assert all([x < type_num for x in vector])
    dis = Dict()
    len = length(vector)
    for (v, w) in zip(vector, weight)
        dis[v] = get(dis, v, 0) + w
    end
    for x in 0 : type_num - 1
        dis[x] = (get(dis, x, 0) + smoothing_factor * type_num) /
            (len + type_num * type_num * smoothing_factor)
    end
    return dis
end

function marginal_distribution(pairwise::Dict, index, type_num)::Dict
    @assert index == 1 || index == 2
    @assert type_num == 2
    # simplified version for binary dataset
    if index == 1
        return Dict(0 => pairwise[(0, 0)] + pairwise[(0, 1)], 1 => pairwise[(1, 0)] + pairwise[(1, 1)])
    else
        return Dict(0 => pairwise[(0, 0)] + pairwise[(1, 0)], 1 => pairwise[(0, 1)] + pairwise[(1, 1)])
    end
end


"calculate pairwise distribution of two weighted array"
function pairwise_distribution(vector1::AbstractArray, vector2::AbstractArray,
        weight::Array, type_num::Int, smoothing_factor::Real)::Dict
    @assert all([x < type_num for x in vector1])
    @assert all([x < type_num for x in vector2])
    @assert length(vector1) == length(vector2)
    dis = Dict()
    len = length(vector1)
    for i in 1 : len
        dis[(vector1[i], vector2[i])] = get(dis, (vector1[i], vector2[i]), 0) + weight[i]
    end
    for x in 0 : type_num - 1, y in 0 : type_num - 1
        dis[(x, y)] = (get(dis, (x, y), 0) + smoothing_factor) /
            (len + type_num * type_num * smoothing_factor)
    end
    return dis
end


"calculate mutual information of two columns of given matrix"
function mutual_information(data::WXData, index1::Int, index2::Int, type_num::Int;
        base=ℯ, smoothing_factor=0)::Float64
    weight = Data.weights(data)
    data_matrix = feature_matrix(data)
    vector1 = data_matrix[:, index1]
    vector2 = data_matrix[:, index2]

    prob_ij = pairwise_distribution(vector1, vector2, weight, type_num, smoothing_factor)
    prob_i = marginal_distribution(prob_ij, 1, type_num)
    prob_j = marginal_distribution(prob_ij, 2, type_num)
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

"get CPTs of Chow-Liu tree, via given data"
function get_cpt(data::WXData, parent_index::Int, child_index::Int,
        type_num::Int;smoothing_factor=0)::Dict
    weight_vector = Data.weights(data)
    data_matrix = feature_matrix(data)
    child = data_matrix[:, child_index]

    if parent_index == 0
        prob_c = marginal_distribution(child, weight_vector, type_num, smoothing_factor)
        return prob_c
    end

    parent = data_matrix[:, parent_index]
    prob_pc = pairwise_distribution(parent, child, weight_vector, type_num, smoothing_factor)
    prob_p = marginal_distribution(prob_pc, 1, type_num)
    prob_c = marginal_distribution(prob_pc, 2, type_num)

    cpt = Dict()
    for p in keys(prob_p), c in keys(prob_c)
        if !isapprox(0.0, prob_p[p]; atol=eps(Float64), rtol=0)
            cpt[(c, p)] = prob_pc[(p, c)] / prob_p[p]
        end
    end
    return cpt
end


#####################
# Learn a Chow-Liu tree from weighted data
#####################
"learn a Chow-Liu tree from data matrix, with Laplace smoothing"
function learn_chow_liu_tree(data::WXData; smoothing_factor=0,num_mix=1,flag=false)
    weight_vector = Data.weights(data)
    data_matrix = feature_matrix(data)
    features_num = num_features(data)
    type_num = 2 #binary dataset

    # Calculate mutual information matrix
    g = SimpleWeightedGraph(features_num)
    for i in 1:features_num, j in i+1:features_num
        mi = mutual_information(data, i, j, type_num;smoothing_factor=smoothing_factor)
        add_edge!(g, i, j, - mi)
    end

    # Maximum spanning tree/ forest
    mst_edges = kruskal_mst(g)
    tree = SimpleWeightedGraph(features_num)
    for edge in mst_edges
        add_edge!(tree, src(edge), dst(edge), - weight(edge))
    end

    # learning trees with differerent roots
    mix_trees = Vector{MetaDiGraph}(undef, num_mix)

    for index in 1 : num_mix

        # Calculate with different roots
        roots = [c[mod(index, length(c)) + 1] for c in connected_components(tree)]
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
        for root in roots set_prop!(clt, root, :parent, 0) end
        for edge in edges(clt)
            set_prop!(clt, dst(edge), :parent, src(edge))
        end
        ## calculate cpts
        for v in vertices(clt)
            parent = get_prop(clt, v, :parent)
            cpt_matrix = get_cpt(data, parent, v, type_num; smoothing_factor = smoothing_factor)
            set_prop!(clt, v, :cpt, cpt_matrix)
        end

        mix_trees[index] = clt
    end

    if flag return mix_trees
    else return mix_trees[1] end
end

"print edges and vertices of a ChowLiu tree"
function print_tree(clt)
    for e in edges(clt) print(e); print(" "); println(props(clt, e)) end
    for v in vertices(clt) print(v); print(" "); println(props(clt, v)) end
end


"calculate complete disjoint probability from Chow-Liu Tree"
function get_log_inference(clt, query)
    @assert(length(query) == nv(clt))
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

"calculate disjoint probability for every sample"
function clt_log_likelihood_per_instance(clt, data)
    data_matrix = feature_matrix(data)
    num_sample = size(data_matrix)[1]
    features_num = num_features(data)
    result = zeros(num_sample)
    for v in vertices(clt)
        parent = get_prop(clt, v, :parent)
        cpt = get_prop(clt, v, :cpt)
        if parent == 0
            result += log.([cpt[data_matrix[i, v]] for i in 1 : num_sample])
        else
            result += log.([cpt[(data_matrix[i, v], data_matrix[i, parent])] for i in 1 : num_sample])
        end
    end
    return result
end
