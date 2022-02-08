using ChowLiuTrees, CUDA
using Graphs: SimpleGraph, SimpleDiGraph, bfs_tree, center, 
connected_components, induced_subgraph, nv, add_edge!
using MetaGraphs: get_prop, set_prop!, MetaDiGraph, vertices, indegree, outneighbors

export hclt


num_categories(d::Matrix) = length(unique(d))
num_categories(d::CuMatrix) = length(unique(Array(d)))


function hclt(data::Union{CuMatrix, Matrix}, ::Type{T} = ProbCircuit;
              latent_heuristic::String = "vanila",
              input_type::Type{<:InputDist} = LiteralDist,
              num_cats::Integer = num_categories(data),
              num_hidden_cats::Integer = 16,
              pseudocount::Float64 = 0.1,
              Float=Float32) where T
    
    num_vars = size(data, 2)

    # Chow-Liu Tree (CLT) given data
    edges = ChowLiuTrees.learn_chow_liu_tree(data; num_trees=1, dropout_prob=0.0, weights=nothing, pseudocount, Float)
    clt = clt_edges2graphs(edges)[1]
    
    # compile hclt from clt
    observed_leafs = categorical_leaves(num_vars, num_cats, input_type, T)
    pc = if latent_heuristic == "vanila"
            hclt_from_clt_vanila(clt::MetaDiGraph, num_cats; num_hidden_cats, leaves=observed_leafs)
        elseif latent_heuristic == "mixed"
            hclt_from_clt_mixed(clt::MetaDiGraph, num_cats; 
                num_max_hidden_cats=num_hidden_cats, leaves=observed_leafs)
        else
            @assert false "Latent heuristic $latent_heuristic can not found."
        end
    
    pc
end


function hclt_from_clt_vanila(clt::MetaDiGraph, num_cats::Integer, ::Type{T} = ProbCircuit;
            num_hidden_cats::Integer = 16, leaves = nothing, input_type::Type{<:InputDist} = LiteralDist) where T
    
    # Circuits representing the leaves
    # hidden_leafs = categorical_leafs(num_vars, num_hidden_cats, T; var_idx_offset)
    
    num_vars = nv(clt)

    if isnothing(leaves)
        leaves = categorical_leaves(num_vars, num_cats, input_type, T)
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
        sub_root = vmap[center(sg)[1]]
        clt = union(clt, bfs_tree(MStree, sub_root))
    end
    MetaDiGraph(clt)
end


function categorical_leaves(num_vars, num_cats, input_type::Union{Type{LiteralDist},Type{BernoulliDist}}, 
                            ::Type{T} = ProbCircuit) where T
    num_bits = num_bits_per_cat(num_cats)

    if input_type == LiteralDist
        plits = input_nodes(ProbCircuit, LiteralDist, num_vars * num_bits; sign = true)
        nlits = input_nodes(ProbCircuit, LiteralDist, num_vars * num_bits; sign = false)
    else
        @assert input_type == BernoulliDist
        plits = input_nodes(ProbCircuit, BernoulliDist, num_vars * num_bits; p = Float32(0.9))
        nlits = input_nodes(ProbCircuit, BernoulliDist, num_vars * num_bits; p = Float32(0.1))
    end
    
    cat_leaf(var, cat) = begin
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


function hclt_from_clt_mixed(clt::MetaDiGraph, num_cats::Integer, ::Type{T} = ProbCircuit;
            num_max_hidden_cats::Integer = 16, leaves = nothing, input_type::Type{<:InputDist} = LiteralDist) where T
    
    # Circuits representing the leaves
    # hidden_leafs = categorical_leafs(num_vars, num_hidden_cats, T; var_idx_offset)
    
    num_vars = nv(clt)

    # Construct the CLT circuit bottom-up    
    node_seq = bottom_up_order_mixed(clt)

    if isnothing(leaves)
        leaves = categorical_leaves(num_vars, num_cats, input_type, T)
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