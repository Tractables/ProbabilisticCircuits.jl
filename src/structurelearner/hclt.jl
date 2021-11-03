export hclt

using DataStructures: PriorityQueue, enqueue!, dequeue!, dequeue_pair!

####################
# Helper functions #
####################

num_bits_for_cats(num_cats) = ceil(Int,log2(num_cats))

num_categories(voc) = count(x -> !iszero(x), voc) 

function as_bits(category, num_bits, bits = BitVector(undef, num_bits))
    for bit_idx = 1:num_bits
        @inbounds bits[bit_idx] = (category & 1) 
        category = category >> 1
    end
    bits
end

function as_cats(bits)
    category = 0
    for bit_idx = length(bits):-1:1
        category = category << 1
        @inbounds category |= bits[bit_idx]
    end
    category
end


function categorical_leafs(num_vars, num_cats, ::Type{T} = ProbCircuit; mode = "pos", var_idx_offset = 0) where T
    num_bits = num_bits_for_cats(num_cats)

    plits = pos_literals(ProbCircuit, num_vars * num_bits)
    nlits = neg_literals(ProbCircuit, num_vars * num_bits)
    
    offset_var_idx(node) = PlainProbLiteralNode(node.literal + sign(node.literal) * var_idx_offset)
    plits = [offset_var_idx(n) for n in plits]
    nlits = [offset_var_idx(n) for n in nlits]
    
    cat_leaf(var,cat) = begin
        bits = as_bits(cat, num_bits)
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


function as_categorical_data(dataset::DataFrame, num_vars, num_cats)
    
    # Convert binary data to categorical data
    if isweighted(dataset)
        dataset, weights = split_sample_weights(dataset)
        weights = convert(Vector{Float64}, weights)
    else
        weights = nothing
    end
    
    num_samples = num_examples(dataset)
    num_bits = ceil(Int,log2(num_cats))
    
    # Get categorical dataset from the binarized dataset
    categorical_dataset = Matrix{UInt32}(undef, num_samples, num_vars)
    if iscomplete(dataset)
        for sample_idx = 1 : num_samples
            for variable_idx = 1 : num_vars
                @inbounds categorical_dataset[sample_idx, variable_idx] = as_cat(dataset[sample_idx, (variable_idx - 1) * num_bits + 1 : variable_idx * num_bits]; complete = true)
            end
        end
    else # If the dataset contains missing values, we impute the missing values with the mode of each column
        for variable_idx = 1 : num_vars
            cat_counts::Array{UInt32} = zeros(UInt32, num_cats)
            for sample_idx = 1 : num_samples
                category = as_cat(dataset[sample_idx, (variable_idx - 1) * num_bits + 1 : variable_idx * num_bits]; complete = false)
                if category != typemax(UInt32)
                    cat_counts[category] += 1
                end
                @inbounds categorical_dataset[sample_idx, variable_idx] = category
            end
            cat_mode = argmax(cat_counts)
            for sample_idx = 1 : num_samples
                if categorical_dataset[sample_idx, variable_idx] == typemax(UInt32)
                    @inbounds categorical_dataset[sample_idx, variable_idx] = cat_mode
                end
            end
        end
    end
    return categorical_dataset, weights
end


function as_cat(bits; complete)
    if !complete && !iscomplete(bits)
        return typemax(UInt32)
    end
    
    category::UInt32 = UInt32(0)
    for bit_idx = length(bits) : -1 : 1
        category = (category << 1) + bits[bit_idx]
    end
    
    (category == 0) ? 2^(length(bits)) : category
end


"Compute the Chow-Liu Tree given a binary dataset.
 Automatically convert to categorical dataset if specified by `num_vars' and `num_cats'.
 If `num_trees` is greater than 1, the algorithm returns the top-K maximum spanning trees
 with respect to the pairwise_MI weights.
 Reference: Listing all the minimum spanning trees in an undirected graph
 http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf"
function chow_liu_tree(data, num_vars, num_cats; pseudocount = 0.1, num_trees::Integer = 1, dropout_prob::Float64 = 0.0)
    # Compute pairwise mutual information between variables
    MI = pairwise_MI(data, num_vars, num_cats; pseudocount)
    
    # Priority queue that maintain candidate MSTs
    candidates = PriorityQueue{Tuple{Vector{SimpleWeightedEdge}, Vector{SimpleWeightedEdge}, Vector{SimpleWeightedEdge}}, Float32}()
    
    # The fully connect graph and its weight
    g = SimpleWeightedGraph(complete_graph(num_vars))
    weights = -MI
    
    included_edges::Vector{SimpleWeightedEdge} = Vector{SimpleWeightedEdge}()
    excluded_edges::Vector{SimpleWeightedEdge} = Vector{SimpleWeightedEdge}()
    reuse = Matrix{Float64}(undef, num_vars, num_vars)
    topk_msts::Vector{Vector{SimpleWeightedEdge}} = Vector{Vector{SimpleWeightedEdge}}()
    
    # Initialize `candidate` with the global MST
    mst_edges, total_weight = MST(g, weights, included_edges, excluded_edges; reuse, dropout_prob = 0.0)
    enqueue!(candidates, (mst_edges, included_edges, excluded_edges), total_weight)
    
    if Threads.nthreads() == 1
        
        # Sequential code
        for idx = 1 : num_trees
            if isempty(candidates)
                break
            end

            (mst_edges, included_edges, excluded_edges), total_weight = dequeue_pair!(candidates)

            # Record the current ST into `topk_msts`
            push!(topk_msts, mst_edges)
            
            if idx == num_trees
                break
            end

            edge_added = false
            for edge_idx = 1 : length(mst_edges)
                if mst_edges[edge_idx] in included_edges
                    continue
                end

                if edge_added
                    push!(included_edges, pop!(excluded_edges))
                end
                push!(excluded_edges, mst_edges[edge_idx])
                edge_added = true

                candidate_mst, total_weight = MST(g, weights, included_edges, excluded_edges; reuse, dropout_prob)
                if candidate_mst !== nothing
                    # A shallow copy of the vectors `included_edges` and `excluded_edges` is sufficient
                    enqueue!(candidates, (candidate_mst, copy(included_edges), copy(excluded_edges)), total_weight) 
                end
            end
        end
        
    else
        
        # Parallel code
        reuse = map(1:Threads.nthreads()) do idx
            Matrix{Float64}(undef, num_vars, num_vars)
        end
        g = map(1:Threads.nthreads()) do idx
            deepcopy(g)
        end
        weights = map(1:Threads.nthreads()) do idx
            deepcopy(weights)
        end
        
        l = ReentrantLock()
        
        for idx = 1 : num_trees
            if isempty(candidates)
                break
            end

            (mst_edges, included_edges, excluded_edges), total_weight = dequeue_pair!(candidates)

            # Record the current ST into `topk_msts`
            push!(topk_msts, mst_edges)
            
            if idx == num_trees
                break
            end

            Threads.@threads for edge_idx = 1 : length(mst_edges)
                curr_included_edges = copy(included_edges)
                curr_excluded_edges = copy(excluded_edges)
                for edge in mst_edges[1:edge_idx-1]
                    if !(edge in included_edges)
                        push!(curr_included_edges, edge)
                    end
                end
                if !(mst_edges[edge_idx] in excluded_edges)
                    push!(curr_excluded_edges, mst_edges[edge_idx])
                end

                id = Threads.threadid()
                candidate_mst, total_weight = MST(g[id], weights[id], curr_included_edges, curr_excluded_edges; reuse = reuse[id], dropout_prob)

                lock(l)
                if candidate_mst !== nothing
                    # A shallow copy of the vectors `included_edges` and `excluded_edges` is sufficient
                    enqueue!(candidates, (candidate_mst, copy(included_edges), copy(excluded_edges)), total_weight) 
                end
                unlock(l)
            end
        end
        
    end
    
    # Post-process the top-K Spanning Trees
    map(topk_msts) do mst_edges
        MStree = SimpleGraph(num_vars)
        map(mst_edges) do edge
            add_edge!(MStree, src(edge), dst(edge))
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
end

"Return a vertex sequence to traverse the tree `g', where children are accessed before parent."
function bottom_up_traverse_node_seq(g::MetaDiGraph)
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


"Compute the Minimum Spanning Tree (MST) of graph g with weights `weights`, with
 constraints such that `included_edges` should be included while `excluded_edges` 
 should be excluded."
function MST(g::SimpleWeightedGraph, weights::Matrix{T}, 
             included_edges::Vector{SimpleWeightedEdge}, 
             excluded_edges::Vector{SimpleWeightedEdge}; reuse::Matrix{T}, dropout_prob = 0.0) where T <: AbstractFloat
    @inbounds @views reuse[:, :] .= weights[:, :]
    
    # Dropout
    if dropout_prob > 1e-8
        dropped_mask = rand(Bernoulli(dropout_prob), size(reuse, 1), size(reuse, 2))
        @inbounds @views reuse[dropped_mask] .= 10000.0
    end
    
    # Add constraints
    map(included_edges) do edge
        reuse[src(edge), dst(edge)] = -10000.0
        reuse[dst(edge), src(edge)] = -10000.0
        nothing # Return nothing to save some effort
    end
    map(excluded_edges) do edge
        reuse[src(edge), dst(edge)] = 10000.0
        reuse[dst(edge), src(edge)] = 10000.0
        nothing # Return nothing to save some effort
    end
    
    mst_edges = kruskal_mst(g, reuse)
    
    # Senity check
    valid_tree::Bool = true
    
    edges = Set(mst_edges)
    map(included_edges) do edge
        if !(edge in edges)
            valid_tree = false
        end
        nothing
    end
    map(excluded_edges) do edge
        if (edge in edges)
            valid_tree = false
        end
        nothing
    end
    
    if valid_tree
        # Compute the tree weight
        total_weight::T = 0.0
        map(mst_edges) do edge
            total_weight += weights[src(edge), dst(edge)]
            nothing
        end
        mst_edges, total_weight
    else
        nothing, nothing
    end
end

"Compute pairwise Mutual Information given binary/categorical data."
function pairwise_MI(dataset::DataFrame, num_vars, num_cats; pseudocount = 1.0)
    categorical_dataset, weights = as_categorical_data(dataset::DataFrame, num_vars, num_cats)
    pairwise_MI(categorical_dataset, num_vars, num_cats, weights; pseudocount = pseudocount)
end
function pairwise_MI(dataset::Matrix, num_vars, num_cats, weights = nothing; pseudocount = 1.0)
    num_samples = size(dataset, 1)
    num_vars = size(dataset, 2)
    
    if weights === nothing
        weights = ones(Int32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end
    
    # Sum of the weights
    sum_weights::Float64 = Float64(sum(weights) + num_cats^2 * pseudocount)
    
    # `joint_cont[i, j, k, w]' is the total weight of samples whose i- and j-th variable are k and w, respectively
    joint_cont = pairwise_marginals(dataset, weights, num_cats; pseudocount)
    
    # `marginal_cont[i, j]' is the total weight of sample whose i-th variable is j
    marginal_cont = zeros(Float64, num_vars, num_cats)
    for i = 1:num_vars
        for j = 1:num_cats
            @inbounds marginal_cont[i,j] = joint_cont[i,i,j,j]
        end
    end
    
    # Compute mutual information
    MI = zeros(Float64, num_vars, num_vars)
    for var1_idx = 1 : num_vars
        for var2_idx = var1_idx : num_vars
            @inbounds MI[var1_idx, var2_idx] = sum(joint_cont[var1_idx, var2_idx, :, :] .* (@. log(sum_weights .* joint_cont[var1_idx, var2_idx, :, :] / (marginal_cont[var1_idx, :] .* marginal_cont[var2_idx, :]')))) / sum_weights
        end
    end
    
    for var1_idx = 2 : num_vars
        for var2_idx = 1 : var1_idx - 1
            @inbounds MI[var1_idx, var2_idx] = MI[var2_idx, var1_idx]
        end
    end
    
    MI
end


function hclt(num_vars, num_cats = 2, ::Type{T} = ProbCircuit; data::DataFrame, 
              num_hidden_cats::Integer = 16, num_trees::Integer = 1, 
              num_tree_candidates::Integer = 1, tree_sample_type::String = "fixed_interval",
              dropout_prob::Float64 = 0.0) where T
    # Chow-Liu Tree (CLT) given data
    clts = chow_liu_tree(data, num_vars, num_cats; num_trees = num_tree_candidates, dropout_prob)
    
    # Sample `num_trees` trees from the `num_tree_candidates` candidates
    if tree_sample_type == "random"
        clts = clts[randperm(num_tree_candidates)[1:num_trees]]
    elseif tree_sample_type == "fixed_interval"
        clts = clts[Int.(round.(LinRange(1, num_tree_candidates, num_trees)))]
    end
    
    observed_leafs = categorical_leafs(num_vars, num_cats, T)
    var_idx_offset = num_vars * num_bits_for_cats(num_cats)
    circuits = map(clts) do clt
        pc = hclt(clt, num_vars, num_cats, observed_leafs; data = data, num_hidden_cats = num_hidden_cats,
                  var_idx_offset = var_idx_offset)
        var_idx_offset += num_vars * num_bits_for_cats(num_hidden_cats)
        
        pc
    end
    
    children::Array{T} = Array{T}(undef, 0)
    for circuit in circuits
        append!(children, circuit.children)
    end
    summate(children...)
end
function hclt(clt::CLT, num_vars, num_cats, observed_leafs, ::Type{T} = ProbCircuit; 
              data::DataFrame, var_idx_offset::Integer = 0, num_hidden_cats::Integer = 4) where T
    # Circuits representing the leaves
    hidden_leafs = categorical_leafs(num_vars, num_hidden_cats, T; var_idx_offset)
    
    # meaning: `joined_leafs[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    gen_joined_leaf(var_idx, hidden_cat_idx) = begin
        # This line encodes the hidden leafs explicitly
        # summate([multiply(hidden_leafs[var_idx, hidden_cat_idx], observed_leafs[var_idx, i]) for i = 1 : num_cats])
        
        # This line does not encode the hidden leafs
        summate(observed_leafs[var_idx, :])
    end
    joined_leafs = gen_joined_leaf.(1:num_vars, (1:num_hidden_cats)')
    
    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_traverse_node_seq(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `circuits' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `circuits' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
        
        if length(out_neighbors) == 0
            # Leaf node
            # We do not add hidden variables for leaf nodes
            circuits = [summate(observed_leafs[curr_node, :]) for idx = 1 : num_hidden_cats]
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