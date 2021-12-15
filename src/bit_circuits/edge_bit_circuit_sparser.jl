
export SparserEdgeBitProbCircuits, init_mar, init_mar!, eval_circuit!, eval_layer!

module SparserEdgeBitProbCircuits
        
    using DirectedAcyclicGraphs, ProbabilisticCircuits, CUDA

    struct BitsLiteral
        literal::Int
    end

    struct BitsInnerNode
        issum::Bool
    end
    const BitsNode = Union{BitsLiteral, BitsInnerNode}

    struct SumEdge 
        parent_id::Int
        prime_id_1::Int
        sub_id_1::Int # 0 means no sub
        logprob_1::Float32
        prime_id_2::Int # 0 means no second prime
        sub_id_2::Int # 0 means no second sub
        logprob_2::Float32
        sync::Bool
    end

    struct MulEdge 
        parent_id::Int
        prime_id::Int
        sub_id::Int # 0 means no sub
    end

    const BitsEdge = Union{SumEdge,MulEdge}
    const EdgeLayer = Vector{BitsEdge}

    abstract type AbstractBitsProbCircuit end

    struct BitsProbCircuit <: AbstractBitsProbCircuit
        nodes::Vector{BitsNode}
        edge_layers::Vector{EdgeLayer}
        BitsProbCircuit(num_nodes:: Int, num_layers::Int) = new(
            Vector{BitsNode}(undef, num_nodes),
            EdgeLayer[EdgeLayer() for i = 1:num_layers-1]
        )
    end

    nondupplicate(root, n) = if isduplicatenode(root, n) 
            nondupplicate(root, children(n)[1]) 
        else 
            n 
        end

    isduplicatenode(root, n) = begin
       (n !== root) && num_children(n)==1 && ismaterializednode(root, nondupplicate(root, children(n)[1]) )
    end

    ismaterializednode(root, n) = begin
        n===root && return true # always materialize the root
        if (ismul(n) && num_children(n)==2) 
            ndc = nondupplicate(root, children(n)[1])
            !ismaterializednode(root, ndc) && error("Wanted to not materialize $n but its first non-duplicate child is $(ndc) which is not materialized")
            ndc = nondupplicate(root, children(n)[2])
            !ismaterializednode(root, ndc) && error("Wanted to not materialize $n but its second non-duplicate child is $(ndc) which is not materialized")
            return false # do not materialize "elements"
        end
        isduplicatenode(root, n) && return false # do not materialize duplicate pass-through nodes
        return true
    end

    function label_nodes_custom(root)
        labeling = Dict{ProbCircuit,Int}()
        i = 0
        f_inner(n, call) = begin 
            child_ids = map(call, children(n))
            if isduplicatenode(root, n)
                @assert length(child_ids) == 1
                child_ids[1]
            elseif !ismaterializednode(root, n) 
                0 # this node will be collapsed into an edge
            else
                (i += 1)
            end
        end 
        f_leaf(n) = (i += 1)
        foldup(root, f_leaf, f_inner, Int, labeling)
        labeling, i
    end

    function feedforward_layers_custom(root::DAG)
        node2layer = Dict{DAG, Int}()
        f_inner(n, call) = begin
            cl = mapreduce(call, max, children(n))
            ismaterializednode(root, n) ? cl + 1 : cl
        end
        f_leaf(n) = 1
        num_layers = foldup(root, f_leaf, f_inner, Int, node2layer)
        node2layer, num_layers
    end

    function ids_and_layer(pc, c, node2label, node2layer)
        if ismaterializednode(pc, c)
            node2label[c], 0, node2layer[c]
        else
            prime, sub = children(c)
            layerid = max(node2layer[prime], node2layer[sub])
            node2label[prime], node2label[sub], layerid
        end
    end


    function BitsProbCircuit(pc)
        node2label, num_materialized_nodes = label_nodes_custom(pc)
        node2layer, num_layers = feedforward_layers_custom(pc)
        @show num_materialized_nodes num_layers
        bpc = BitsProbCircuit(num_materialized_nodes, num_layers)
        foreach(pc) do node 
            pid = node2label[node]
            if ismaterializednode(pc, node)
                if isleaf(node)
                    bnode = BitsLiteral(literal(node))
                else
                    child_nodes = children(node)
                    if issum(node)
                        bnode = BitsInnerNode(true)
                        sync = (length(child_nodes) > 2)
                        for i = 1:2:length(child_nodes)
                            logp_1 = node.log_probs[i]
                            primeid_1, subid_1, layerid_1 = ids_and_layer(pc, child_nodes[i], node2label, node2layer)
                            if i == length(child_nodes)
                                logp_2 = 0.0                           
                                primeid_2 = subid_2 = layerid_2 = 0     
                            else
                                logp_2 = node.log_probs[i+1]
                                primeid_2, subid_2, layerid_2 = ids_and_layer(pc, child_nodes[i+1], node2label, node2layer)
                            end
                            edge = SumEdge(pid, primeid_1, subid_1, logp_1, primeid_2, subid_2, logp_2, sync)
                            layer = bpc.edge_layers[max(layerid_1, layerid_2)]
                            push!(layer, edge)
                        end
                    else
                        @assert ismul(node)
                        bnode = BitsInnerNode(false)
                        for c in child_nodes
                            primeid, subid, layerid = ids_and_layer(pc, c, node2label, node2layer)
                            edge = MulEdge(pid, primeid, subid)
                            layer = bpc.edge_layers[layerid]
                            push!(layer, edge) 
                        end
                    end
                end
                bpc.nodes[pid] = bnode
            end
        end
        bpc, node2label
    end

    num_edge_layers(bpc::AbstractBitsProbCircuit) = 
        length(bpc.edge_layers)


    const CuEdgeLayer = CuVector{BitsEdge}

    struct CuProbCircuit <: AbstractBitsProbCircuit
        nodes::CuVector{BitsNode}
        edge_layers::Vector{CuEdgeLayer}
        CuProbCircuit(bpc::BitsProbCircuit) = begin
            nodes = cu(bpc.nodes)
            edge_layers = map(cu, bpc.edge_layers)
            new(nodes, edge_layers)
        end
    end

    function logsumexp(x::Float32,y::Float32)::Float32
        if x == -Inf32
            y
        elseif y == -Inf32
            x
        elseif x > y
            x + log1p(exp(y-x))
        else
            y + log1p(exp(x-y))
        end 
    end
    

end


const SparserEdgeProbCircuit = Union{SparserEdgeBitProbCircuits.BitsProbCircuit, SparserEdgeBitProbCircuits.CuProbCircuit}

#########################################################
## Helpers
#########################################################

function bit_node_stats_sparser(nodes::AbstractVector)
    groups = DirectedAcyclicGraphs.groupby(nodes) do n
        if n isa SparserEdgeBitProbCircuits.SumEdge
            "Sum-$((n.prime_id_1 > 0) + (n.sub_id_1 > 0))-$((n.prime_id_2 > 0) + (n.sub_id_2 > 0))-$(n.sync)"
        else
            @assert n isa SparserEdgeBitProbCircuits.MulEdge
            "Mul-$((n.prime_id > 0) + (n.sub_id > 0))"
        end
    end
    DirectedAcyclicGraphs.map_values(v -> length(v), groups, Int)
end

bit_node_stats(bpc::SparserEdgeBitProbCircuits.BitsProbCircuit) =
    bit_node_stats_sparser(reduce(vcat, bpc.edge_layers))

#############################################################
###### Marginals
#############################################################

init_mar(node::SparserEdgeBitProbCircuits.BitsInnerNode, data, example_id) = 
    node.issum ? -Inf32 : zero(Float32)

function init_mar(leaf::SparserEdgeBitProbCircuits.BitsLiteral, data, example_id)
    lit = leaf.literal
    v = data[abs(lit), example_id]
    if ismissing(v)
        zero(Float32)
    elseif (lit > 0) == v
        zero(Float32)
    else
        -Inf32
    end
end

function init_mar!(mars, bpc::SparserEdgeProbCircuit, data, example_ids)
    broadcast!(mars, bpc.nodes, transpose(example_ids)) do node, example_id
        init_mar(node, data, example_id)
    end
end

function logincexp(matrix::CuDeviceArray, i, j, v, sync)
    if sync
        CUDA.@atomic matrix[i,j] = SparserEdgeBitProbCircuits.logsumexp(matrix[i,j], v)
    else
        matrix[i,j] = SparserEdgeBitProbCircuits.logsumexp(matrix[i,j], v)
    end
    nothing
end

function logincexp(matrix::Array, i, j, v, sync)
    # should also be atomic for CPU multiprocessing?
    matrix[i,j] = SparserEdgeBitProbCircuits.logsumexp(matrix[i,j], v)
    nothing
end

function logmulexp(matrix::CuDeviceArray, i, j, v)
    CUDA.@atomic matrix[i,j] += v
    nothing
end

function logmulexp(matrix::Array, i, j, v)
    # should also be atomic for CPU multiprocessing?
    matrix[i,j] += v
    nothing
end

function eval_edge!(mars, edge::SparserEdgeBitProbCircuits.SumEdge, example_id)
    # first child
    child_prob_1 = mars[edge.prime_id_1, example_id]
    if edge.sub_id_1 > 0
        child_prob_1 += mars[edge.sub_id_1, example_id]
    end
    edge_prob = child_prob_1 + edge.logprob_1
    # second child
    if edge.prime_id_2 > 0
        child_prob_2 = mars[edge.prime_id_2, example_id]
        if edge.sub_id_2 > 0
            child_prob_2 += mars[edge.sub_id_2, example_id]
        end
        edge_prob_2 = child_prob_2 + edge.logprob_2
        edge_prob = SparserEdgeBitProbCircuits.logsumexp(edge_prob, edge_prob_2)
    end
    # increment node value
    logincexp(mars, edge.parent_id, example_id, edge_prob, edge.sync)
    nothing
end

function eval_edge!(mars, edge::SparserEdgeBitProbCircuits.MulEdge, example_id)
    child_prob = mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    logmulexp(mars, edge.parent_id, example_id, child_prob)
    nothing
end

function eval_layer!(mars::Array, bpc::SparserEdgeProbCircuit, layer_id)
    for edge in bpc.edge_layers[layer_id]
        for example_id in 1:size(mars,2)
            eval_edge!(mars, edge, example_id)
        end
    end
    nothing
end

function eval_layer!_kernel_one_edge(mars, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_id = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    ex_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if ex_id <= size(mars,2) && edge_id <= length(layer)
        eval_edge!(mars, layer[edge_id], ex_id)
    end
    nothing
end

function eval_layer!_kernel_edge_loop(mars, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    edge_end = min(edge_start + edge_work - 1, length(layer))

    ex_id = ((blockIdx().y - 1) * blockDim().y + threadIdx().y - 1) + 1
    if ex_id <= size(mars,2)
        for edge_id = edge_start:edge_end
            eval_edge!(mars, layer[edge_id], ex_id)
        end
    end
    nothing
end

function balance_threads_sparser(num_nodes, num_examples, config; block_multiplier)
    # prefer to assign threads to examples, they do not require memory synchronization
    ex_threads = min(config.threads, num_examples)
    # make sure each thread deals with at most one example
    ex_blocks = cld(num_examples, ex_threads)
    node_threads = config.threads ÷ ex_threads
    node_blocks = min(cld(block_multiplier * config.blocks, ex_blocks), cld(num_nodes, node_threads))
    ((node_threads, ex_threads), (node_blocks, ex_blocks))
end

function eval_layer!(mars, bpc, layer_id; block_multiplier=1000, debug=false)
    layer = bpc.edge_layers[layer_id]
    kernel = @cuda name="eval_layer!" launch=false eval_layer!_kernel_edge_loop(mars, layer) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads_sparser(length(layer), size(mars,2), config; block_multiplier)
    debug && println("Layer $layer_id")
    debug && @show config, threads, blocks
    debug && println("Each thread processes $(Float32(length(layer)/threads[1]/blocks[1])) edges")
    if length(layer) <= threads[1]*blocks[1]
        debug && println("Per-edge kernel launch")
        @cuda threads=threads blocks=blocks eval_layer!_kernel_one_edge(mars, layer) 
    else
        debug && println("General kernel launch")
        kernel(mars, layer; threads, blocks)
    end
    nothing
end

# run entire circuit
function eval_circuit!(mars, bpc::SparserEdgeProbCircuit, data, example_ids; block_multiplier=1000, debug=false)
    init_mar!(mars, bpc, data, example_ids)
    for i in 1:SparserEdgeBitProbCircuits.num_edge_layers(bpc)
        eval_layer!(mars, bpc, i; block_multiplier, debug)
    end
    nothing
end
