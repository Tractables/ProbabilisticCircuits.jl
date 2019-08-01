using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

#####################
# Data structure and methods to calculate and compute distribution, applied to binary dataset
#####################

const PairwiseDis = Dict{Tuple{Int64, Int64}, Float64}
const MarginalDis = Dict{Int64, Float64}

mutable struct DisCache{P<:Dict{Tuple{Int64, Int64}, PairwiseDis}, M<:Dict{Int64, MarginalDis}}
    pairwises::P
    marginals::M
end

DisCache() = DisCache(Dict{Tuple{Int64, Int64}, PairwiseDis}(), Dict{Int64, MarginalDis}())

@inline function change_variable(p::PairwiseDis)
    ret = copy(p)
    ret[(0, 1)], ret[(1, 0)] = ret[(1, 0)], ret[(0, 1)]
    return ret
end
@inline get_distribution(index::Tuple{Int64, Int64}, dis_cache::DisCache) = 
    index[1] < index[2] ? dis_cache.pairwises[index] : change_variable(dis_cache.pairwises[(index[2], index[1])])
@inline get_distribution(index::Int64, dis_cache::DisCache) = dis_cache.marginals[index]

"calulate pairwise and marginal for all features"
function calculate_all_distributions(data::WXData, dis_cache::DisCache; type_num = 2, α = 0)
    features_num = num_features(data)
    for i in 1 : features_num, j in i + 1 : features_num
        calculate_distribution((i, j), data, dis_cache; type_num = type_num, α = α)
    end
    for i in 1 : features_num
        calculate_distribution(i, data, dis_cache; type_num = type_num, α = α)
    end
end

calculate_distribution(index::Tuple{Int64, Int64}, data::WXData, dis_cache::DisCache; type_num = 2, α = 0) = 
    calculate_distribution(index, feature_matrix(data), Data.weights(data), dis_cache; type_num = type_num, α = α)

calculate_distribution(index::Tuple{Int64, Int64}, data_matrix::AbstractMatrix, weight::Vector{Float64}, dis_cache::DisCache; type_num = 2, α = 0) = 
    get!(dis_cache.pairwises, index) do
        pairwise_distribution(view(data_matrix, :, index[1]), view(data_matrix, :, index[2]), weight; type_num = type_num, α = α)
    end

calculate_distribution(index::Int64, data::WXData, dis_cache::DisCache; type_num = 2, α = 0) =
    get!(dis_cache.marginals, index) do
        marginal_distribution(index, dis_cache; type_num = type_num, α = α)
    end
    
"calculate marginal distribution of weighted array"
function marginal_distribution(index::Int64, dis_cache::DisCache; type_num = 2, α = 0)::MarginalDis
    dis = MarginalDis()
    # calculate marginal from pairwise
    #if (1, index) in keys(dis_cache.marginals)
    @assert !isempty(dis_cache.pairwises)
    if index == 1
        pairwise = dis_cache.pairwises[(1, 2)]
        dis[0] = pairwise[(0, 0)] + pairwise[(0, 1)]
        dis[1] = pairwise[(1, 0)] + pairwise[(1, 1)]
        return dis
    else
        pairwise = dis_cache.pairwises[(1, index)]
        dis[0] = pairwise[(0, 0)] + pairwise[(1, 0)]
        dis[1] = pairwise[(0, 1)] + pairwise[(1, 1)]
    return dis
    # calculate marginal from original data, be called when node has no neighbors
    #else
    #    weight = Data.weights(data)
    #    data_matrix = feature_matrix(data)
    #    vector = view(data_matrix, :, index)
    #    base = sum(weight)
    #    for (v, w) in zip(vector, weight)
    #        dis[v] = get(dis, v, 0) + w
    #    end
    #    for x in 0 : type_num - 1
    #        dis[x] = (get(dis, x, 0) + α) / (base + type_num * α)
    #    end
    #    return dis
    #end
    end
end

"calculate pairwise distribution of two weighted array"
function pairwise_distribution(vector1::AbstractVector, vector2::AbstractVector, weight::Vector{Float64}; type_num = 2, α = 0)::PairwiseDis
    dis = PairwiseDis()
    base = sum(weight)
    for i in 1 : length(vector1)
        dis[(vector1[i], vector2[i])] = get(dis, (vector1[i], vector2[i]), 0) + weight[i]
    end
    for x in 0 : type_num - 1, y in 0 : type_num - 1
        dis[(x, y)] = (get(dis, (x, y), 0) + α) / (base + type_num * type_num * α)
    end
    return dis
end

#####################
# Methods for CPTs and MIs
#####################

"calculate mutual information of two columns of given matrix"
function mutual_information(index1::Int64, index2::Int64, dis_cache::DisCache)::Float64
    prob_ij = get_distribution((index1, index2), dis_cache)
    prob_i = get_distribution(index1, dis_cache)
    prob_j = get_distribution(index2, dis_cache)
    
    mi = 0.0
    for x in keys(prob_i), y in keys(prob_j)
        if !isapprox(0.0, prob_ij[(x, y)]; atol=eps(Float64), rtol=0)
            mi += prob_ij[(x, y)] * log(prob_ij[(x, y)] / (prob_i[x] * prob_j[y]))
        end
    end
    return mi
end

"get CPTs of Chow-Liu tree, via given data"
function get_cpt(parent_index::Int64, child_index::Int64, dis_cache::DisCache)::Dict
    
    prob_c = get_distribution(child_index, dis_cache)

    if parent_index == 0 return prob_c end

    prob_pc = get_distribution((parent_index, child_index), dis_cache)
    prob_p = get_distribution(parent_index, dis_cache)

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
function learn_chow_liu_tree(data::WXData; α = 0)
    weight_vector = Data.weights(data)
    data_matrix = feature_matrix(data)
    features_num = num_features(data)
    type_num = 2 #binary dataset
    dis_cache = DisCache()

    #Calculate all distrubutions for future use
    calculate_all_distributions(data, dis_cache; type_num = type_num, α = α)

    # Calculate mutual information matrix
    g = SimpleWeightedGraph(features_num)
    for i in 1 : features_num, j in i + 1 : features_num
        mi = mutual_information(i, j, dis_cache)
        add_edge!(g, i, j, - mi)
    end

    # Maximum spanning tree/ forest
    mst_edges = kruskal_mst(g)
    tree = SimpleWeightedGraph(features_num)
    for edge in mst_edges
        add_edge!(tree, src(edge), dst(edge), - weight(edge))
    end


    # Calculate with different roots
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
    for root in roots set_prop!(clt, root, :parent, 0) end
    for edge in edges(clt)
        set_prop!(clt, dst(edge), :parent, src(edge))
    end

    ## calculate cpts
    for v in vertices(clt)
        parent = get_prop(clt, v, :parent)
        cpt_matrix = get_cpt(parent, v, dis_cache)
        set_prop!(clt, v, :cpt, cpt_matrix)
    end
    return clt

end

#####################
# Other methods about CLT
#####################
"compare if the **structure** of two CLTs are equal"
function compare_tree(clt1, clt2)

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
