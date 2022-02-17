using CUDA
using ChowLiuTrees: learn_chow_liu_tree
using Graphs: SimpleGraph, SimpleDiGraph, bfs_tree, center, neighbors,
connected_components, induced_subgraph, nv, add_edge!, rem_edge!
using MetaGraphs: get_prop, set_prop!, MetaDiGraph, vertices, indegree, outneighbors

export hclt

"""
    hclt(data, num_hidden_cats; num_cats = nothing, input_type = LiteralDist)

Learns HiddenChowLiuTree (hclt) circuit structure from data.
- `data`: Matrix or CuMatrix
- `num_hidden_cats`: Number of categories in hidden variables
- `input_type`: Distribution type for the inputs
- `num_cats`: Number of categories (in case of categorical inputs). Automatically deduced if not given explicilty.
"""
function hclt(data, num_hidden_cats; 
              num_cats = nothing,
              shape = :directed,
              input_type = Literal,
              pseudocount = 0.1) where T
    
    clt_edges = learn_chow_liu_tree(data; pseudocount, Float=Float32)
    clt = clt_edges2graphs(clt_edges; shape)
    
    if num_cats === nothing
        num_cats = maximum(data) + 1
    end
    hclt_from_clt(clt, num_cats, num_hidden_cats; input_type)
end


function hclt_from_clt(clt, num_cats, num_hidden_cats; input_type = Literal)
    
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

function clt_edges2graphs(edgepair; shape=:directed)
    vars = sort(collect(Set(append!(first.(edgepair), last.(edgepair)))))
    @assert all(vars .== collect(1:maximum(vars))) "Variables are not contiguous"
    
    nvar = length(vars)
    MStree = SimpleGraph(nvar)
    map(edgepair) do edge
        add_edge!(MStree, edge[1], edge[2])
    end

    if shape == :directed
        # Use the graph center of `MStree` as the root node
        MetaDiGraph(bfs_tree(MStree, center(MStree)[1]))

    elseif shape == :balanced
        # iteratively pick the graph center to make a balanced clt
        clt = SimpleDiGraph(nvar)
        
        function find_center_ite(g, vmap, clt_map, clt) 
            # `vmap` map current `g` index to upper layer graph id
            # `clt_map` map sub graph to `clt`
            # return root
    
            if nv(g) == 1
                return vmap[1]
            else
                root = center(g)[1]
                for dst in collect(neighbors(g, root))
                    rem_edge!(g, root, dst)
                    sub_nodes = filter(x -> dst in x, connected_components(g))
                    add_edge!(g, root, dst)
                    sub_g, sub_vmap = induced_subgraph(g, sub_nodes[1])
                    sub_root = find_center_ite(sub_g, sub_vmap, clt_map[sub_vmap], clt)
                    add_edge!(clt, clt_map[root], clt_map[sub_root])
                end
                return vmap[root]
            end
        end

        find_center_ite(MStree, collect(1:nvar), collect(1:nvar), clt)
        MetaDiGraph(clt)
    else
        error("Shape $shape not found in function `clt_edges2graphs`.")
    end
end


function categorical_leaves(num_vars, num_cats, num_hidden_cats, 
                            input_type::Type{Bernoulli})
    
    @assert num_cats == 2 "Category must be two when leaf node is bernoulli."
    error("TODO: implement way of replacing sum nodes by Berns")
end


function categorical_leaves(num_vars, num_cats, num_hidden_cats, 
                            input_type::Type{Literal})
    if num_cats == 2
        plits = [PlainInputNode(var, Literal(true)) for var=1:num_vars]
        nlits = [PlainInputNode(var, Literal(false)) for var=1:num_vars]
        leaves = hcat([plits, nlits]...)
        [summate(leaves[var, :]) 
            for var=1:num_vars, copy=1:num_hidden_cats]
    else # Use Literal to model categorical distributions
        nbits = Int(ceil(log2(num_cats)))
        plits = [PlainInputNode((var-1)*nbits+lit, Literal(true)) 
                    for var=1:num_vars, lit=1:nbits]
        nlits = [PlainInputNode((var-1)*nbits+lit, Literal(false))
                    for var=1:num_vars, lit=1:nbits]
        to_bits(cat, nbits) = begin
            bits = zeros(Bool, nbits)
            for b = 1 : nbits
                bits[nbits-b+1] = ((cat % 2) == 1)
                cat = cat ÷ 2
            end
            bits
        end
        cat_leaf(var, _) = begin
            cat_lits = map(1:num_cats) do cat
                bits = to_bits(cat, nbits)
                lits = [ifelse(bits[l], plits[var,l], nlits[var,l]) for l=1:nbits]
                multiply(lits...)
            end
            summate(cat_lits...)
        end
        cat_leaf.(1:num_vars, (1:num_hidden_cats)')
    end
end


function categorical_leaves(num_vars, num_cats, num_hidden_cats, input_type::Type{Categorical})
    [PlainInputNode(var, Categorical(num_cats)) 
        for var=1:num_vars, copy=1:num_hidden_cats]
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