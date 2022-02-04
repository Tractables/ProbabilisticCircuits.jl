using Graphs: SimpleGraph, SimpleDiGraph, bfs_tree, center, 
connected_components, induced_subgraph, nv
using MetaGraphs: get_prop, set_prop!, MetaDiGraph
using DataStructures
using ChowLiuTrees

export hclt

using LogicCircuits
using LogicCircuits.Utils: eachcol_unweighted

num_categories(d::Matrix) = length(unique(d))
num_categories(d::CuMatrix) = length(unique(d))
num_categories(d::DataFrame) = length(mapreduce(unique, union, eachcol_unweighted(d)))


function hclt(data::Union{CuMatrix, Matrix}, ::Type{T} = ProbCircuit;
              latent_heuristic::String = "vanila",
              num_cats::Integer = num_categories(data),
              num_hidden_cats::Integer = 16,
              tree_sample_type::String = "fixed_interval", # "random"
              num_trees::Integer = 1, 
              num_tree_candidates::Integer = 1, 
              dropout_prob::Float64 = 0.0,
              weights::Union{Vector, CuVector, Nothing} = nothing, 
              pseudocount::Float64 = 0.1,
              Float=Float32) where T
    
    num_vars = size(data, 2) # num_features(data) # not supported for CuMatrix

    # Chow-Liu Tree (CLT) given data
    # println("Learning a CLT")
    edges = ChowLiuTrees.learn_chow_liu_tree(data;
            num_trees=num_tree_candidates, 
            dropout_prob, weights, pseudocount, Float)
    clts = clt_edges2graphs(edges)
    
    # Sample `num_trees` trees from the `num_tree_candidates` candidates
    if tree_sample_type == "random"
        clts = clts[randperm(num_tree_candidates)[1:num_trees]]
    elseif tree_sample_type == "fixed_interval"
        clts = clts[Int.(round.(LinRange(1, num_tree_candidates, num_trees)))]
    end
    
    # compile hclt from clt
    # println("Compiling to HCLT")
    observed_leafs = categorical_leaves(num_vars, num_cats, T)
    circuits = map(clts) do clt
        if latent_heuristic == "vanila"
            hclt_from_clt_vanila(clt::MetaDiGraph, num_cats; num_hidden_cats, leaves=observed_leafs)
        elseif latent_heuristic == "mixed"
            hclt_from_clt_mixed(clt::MetaDiGraph, num_cats; 
                num_max_hidden_cats=num_hidden_cats, leaves=observed_leafs)
        else
            @assert false "Latent heuristic $latent_heuristic can not found."
        end
    end
    
    children::Array{T} = Array{T}(undef, 0)
    for circuit in circuits
        append!(children, circuit.children)
    end
    summate(children...)
end


function hclt_from_clt_vanila(clt::MetaDiGraph, num_cats::Integer, ::Type{T}=ProbCircuit;
            num_hidden_cats::Integer = 16, leaves=nothing) where T
    
    # Circuits representing the leaves
    # hidden_leafs = categorical_leafs(num_vars, num_hidden_cats, T; var_idx_offset)
    
    num_vars = nv(clt)

    if isnothing(leaves)
        leaves = categorical_leaves(num_vars, num_cats, T)
    end
    
    # meaning: `joined_leafs[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    gen_joined_leaf(var_idx, _) = begin
        # This line encodes the hidden leafs explicitly
        # summate([multiply(hidden_leafs[var_idx, hidden_cat_idx], observed_leafs[var_idx, i]) for i = 1 : num_cats])
        
        # This line does not encode the hidden leafs
        summate(leaves[var_idx, :])
    end
    joined_leafs = gen_joined_leaf.(1:num_vars, (1:num_hidden_cats)')
    
    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_order(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `circuits' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `circuits' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
        
        if length(out_neighbors) == 0
            # Leaf node
            # We do not add hidden variables for leaf nodes
            circuits = [summate(leaves[curr_node, :]) for idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        else
            # Inner node
            
            # Each element in `child_circuits' represents the joint distribution of the child nodes, 
            # i.e., Pr(X_1)...Pr(X_k)
            child_circuits = [get_prop(clt, child_node, :circuits) for child_node in out_neighbors]
            if length(out_neighbors) > 1
                child_circuits = [summate(multiply([child_circuit[cat_idx] for child_circuit in child_circuits])) for cat_idx = 1 : num_hidden_cats]
            else
                child_circuits = child_circuits[1]
            end
            # Pr(X_1)...Pr(X_k) -> Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
            circuits = [summate(multiply.(child_circuits, joined_leafs[curr_node, :])) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node
end


clt_edges2graphs(edges::Vector{<:Vector}) = map(edges) do e
        clt_edges2graphs(e)
end


function clt_edges2graphs(edges::Vector{Tuple{Int,Int}})
    vars = sort(collect(Set(append!(first.(edges), last.(edges)))))
    @assert all(vars .== collect(1:maximum(vars))) "Variables are not contiguous"
    num_vars = length(vars)
    
    MStree = SimpleGraph(num_vars)
    map(edges) do edge
        add_edge!(MStree, edge[1], edge[2])
    end
    
    # Use the graph center of `MStree' as the root node of the CLT
    clt = SimpleDiGraph(num_vars)
    for c in filter(c -> (length(c) > 1), connected_components(MStree))
        sg, vmap = induced_subgraph(MStree, c)
        sub_root = vmap[Graphs.center(sg)[1]]
        clt = union(clt, bfs_tree(MStree, sub_root))
    end
    MetaDiGraph(clt)
end


function categorical_leaves(num_vars, num_cats, ::Type{T} = ProbCircuit) where T
    num_bits = num_bits_per_cat(num_cats)

    plits = pos_literals(ProbCircuit, num_vars * num_bits)
    nlits = neg_literals(ProbCircuit, num_vars * num_bits)
    
    offset_var_idx(node) = PlainProbLiteralNode(node.literal)
    plits = [offset_var_idx(n) for n in plits]
    nlits = [offset_var_idx(n) for n in nlits]
    
    cat_leaf(var,cat) = begin
        bits = to_bits(cat, num_bits)
        binary_leafs = Vector{T}(undef, num_bits)
        for i = 1 : num_bits
            bit_index = (var-1) * num_bits + i
            lit = bits[i] ? plits[bit_index] : nlits[bit_index]
            binary_leafs[i] = lit
        end
        if num_bits >= 2
            multiply(binary_leafs...)
        else
            binary_leafs[1]
        end
    end

    cat_leaf.(1:num_vars, (1:num_cats)')
end


num_bits_per_cat(num_cats) = ceil(Int, log2(num_cats))


function to_bits(category, num_bits, bits = BitVector(undef, num_bits))
    for bit_idx = 1:num_bits
        @inbounds bits[bit_idx] = (category & 1) 
        category = category >> 1
    end
    bits
end


function bottom_up_order(g::MetaDiGraph)
    num_nodes = length(vertices(g))
    node_seq = Array{UInt32, 1}(undef, num_nodes)
    idx = 1
    
    function dfs(node_idx)
        out_neighbors = outneighbors(g, node_idx)
        
        for out_neighbor in out_neighbors
            dfs(out_neighbor)
        end
        
        node_seq[idx] = node_idx
        idx += 1
    end
        
    root_node_idx = findall(x->x==0, indegree(g))[1]
    dfs(root_node_idx)
    
    @assert idx == num_nodes + 1
    
    node_seq
end


function hclt_from_clt_mixed(clt::MetaDiGraph, num_cats::Integer, ::Type{T}=ProbCircuit;
            num_max_hidden_cats::Integer = 16, leaves=nothing) where T
    
    # Circuits representing the leaves
    # hidden_leafs = categorical_leafs(num_vars, num_hidden_cats, T; var_idx_offset)
    
    num_vars = nv(clt)

    # Construct the CLT circuit bottom-up    
    node_seq = bottom_up_order_mixed(clt)

    if isnothing(leaves)
        leaves = categorical_leaves(num_vars, num_cats, T)
    end
    
    # meaning: `joined_leafs[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    gen_joined_leaf(var_idx, j) = begin
        # This line encodes the hidden leafs explicitly
        # summate([multiply(hidden_leafs[var_idx, hidden_cat_idx], observed_leafs[var_idx, i]) for i = 1 : num_cats])
        
        # This line does not encode the hidden leafs
        x = summate(leaves[var_idx, :])
        x
    end


    joined_leafs = map(1:num_vars) do var 
        # gen_joined_leaf.(1:num_vars, num_hidden_leafs)
        num_copy = minimum([num_cats ^ get_prop(clt, var, :height), num_max_hidden_cats])
        # println(var, " ", num_copy)
        [gen_joined_leaf(var, j) for j in 1:num_copy]
    end
    

    for curr_node in node_seq
        num_cat_cur = minimum([num_cats ^ get_prop(clt, curr_node, :height), num_max_hidden_cats])
        num_copy = minimum([num_cats ^ get_prop(clt, curr_node, :parent_height), num_max_hidden_cats])
        
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `circuits' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `circuits' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
        
        if length(out_neighbors) == 0
            # Leaf node
            # We do not add hidden variables for leaf nodes
            circuits = [summate(leaves[curr_node, :]) for idx = 1 : num_copy]
            set_prop!(clt, curr_node, :circuits, circuits)

        else
            # Inner node
            
            # Each element in `child_circuits' represents the joint distribution of the child nodes, 
            # i.e., Pr(X_1)...Pr(X_k)
            child_circuits = [get_prop(clt, child_node, :circuits) for child_node in out_neighbors]
            if length(out_neighbors) > 1
                child_circuits = [summate(multiply([child_circuit[cat_idx] for child_circuit in child_circuits])) for cat_idx = 1 : num_cat_cur]
            else
                child_circuits = child_circuits[1]
            end
            # Pr(X_1)...Pr(X_k) -> Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
            circuits = [summate(multiply.(child_circuits, joined_leafs[curr_node])) for cat_idx = 1 : num_copy]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node
end


function bottom_up_order_mixed(g::MetaDiGraph)
    num_nodes = length(vertices(g))
    node_seq = Array{UInt32, 1}(undef, num_nodes)
    idx = 1

    # map node_seq to num_cats for each variable
    
    function dfs(node_idx)
        out_neighbors = outneighbors(g, node_idx)
        
        height = nothing
        if length(out_neighbors) == 0
            height = 0
        else
            heights = map(out_neighbors) do out_neighbor
                dfs(out_neighbor)
            end
            height = maximum(heights) + 1
        end
        
        node_seq[idx] = node_idx
        idx += 1

        set_prop!(g, node_idx, :height, height)
        return height
    end
        
    root_node_idx = findall(x->x==0, indegree(g))[1]
    dfs(root_node_idx)
    
    @assert idx == num_nodes + 1
    
    # parent height is the number of copy
    for n in vertices(g)
        parent = inneighbors(g, n)
        if length(parent) == 0
            set_prop!(g, n, :parent_height, 0) # root
        else
            @assert length(parent) == 1
            parent = parent[1]
            height = get_prop(g, parent, :height)
            set_prop!(g, n, :parent_height, height)
        end
    end
    node_seq
end


