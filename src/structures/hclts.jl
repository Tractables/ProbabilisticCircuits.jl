using CUDA
using ChowLiuTrees: learn_chow_liu_tree
using Graphs: SimpleGraph, SimpleDiGraph, bfs_tree, center, 
connected_components, induced_subgraph, nv, add_edge!
using MetaGraphs: get_prop, set_prop!, MetaDiGraph, vertices, indegree, outneighbors

export hclt

function hclt(data, num_hidden_cats;
              input_type = LiteralDist,
              pseudocount = 0.1) where T
    
    clt_edges = learn_chow_liu_tree(data; pseudocount, Float=Float32)
    clt = clt_edges2graphs(clt_edges)
    
    num_cats = maximum(data) - minimum(data) + 1
    hclt_from_clt(clt, num_cats, num_hidden_cats; input_type)
end


function hclt_from_clt(clt, num_cats, num_hidden_cats; input_type = LiteralDist)
    
    num_vars = nv(clt)

    # meaning: `joined_leaves[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    joined_leaves = categorical_leaves(num_vars, num_cats, num_hidden_cats, input_type)
    
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
            circuits = joined_leaves[curr_node, :]
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
            circuits = [summate(multiply.(child_circuits, joined_leaves[curr_node, :])) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node
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


function categorical_leaves(num_vars, num_cats, num_hidden_cats, 
                            input_type::Union{Type{LiteralDist},Type{BernoulliDist}})
    num_bits = num_bits_per_cat(num_cats)

    if input_type == LiteralDist
        plits = [PlainInputNode(var, LiteralDist(true)) for var=1:num_vars * num_bits]
        nlits = [PlainInputNode(var, LiteralDist(false)) for var=1:num_vars * num_bits]
    else
        @assert input_type == BernoulliDist
        error("TODO: implement way of replacing sum nodes by Berns")
    end
    
    cat_leaf(var, cat) = begin
        # TODO remove this conversion, no longer needed
        bits = to_bits(cat, num_bits)
        binary_leafs = Vector{PlainInputNode}(undef, num_bits)
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

    leaves = [cat_leaf(v,c) for v=1:num_vars, c=1:num_cats]
    [summate(leaves[var, :]) 
        for var=1:num_vars, copy=1:num_hidden_cats]
end

function categorical_leaves(num_vars, num_cats, num_hidden_cats, input_type::Type{CategoricalDist})
    [PlainInputNode(var, CategoricalDist(num_cats)) 
        for var=1:num_vars, copy=1:num_hidden_cats]
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