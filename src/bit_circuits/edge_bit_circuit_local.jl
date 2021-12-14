export bit_node_stats, init_mar, init_mar!, eval_circuit!, eval_layer!


module LocalEdgeBitProbCircuits
        
    using DirectedAcyclicGraphs, ProbabilisticCircuits, CUDA

    struct BitsLiteral
        literal::Int
    end

    struct BitsInnerNode
        issum::Bool
    end
    const BitsNode = Union{BitsLiteral, BitsInnerNode}

    struct SumEdge 
        parent_id::UInt32
        prime_id::UInt32
        sub_id::UInt32 # 0 means no sub
        logp::Float32
        first_or_last::UInt8
    end

    struct MulEdge 
        parent_id::UInt32
        prime_id::UInt32
        sub_id::UInt32 # 0 means no sub
        first_or_last::UInt8
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

    function prime_sub_ids(pc, c, node2label)
        if ismaterializednode(pc, c)
            node2label[c], 0
        else
            prime, sub = children(c)
            node2label[prime], node2label[sub]
        end
    end

    first_or_last(i,n) =
        (i == n == 1) ? 0 : (i == 1) ? 1 : (i<n) ? 2 : 3 

    function BitsProbCircuit(pc)
        node2label, num_materialized_nodes = label_nodes_custom(pc)
        node2layer, num_layers = feedforward_layers_custom(pc)
        #@show num_materialized_nodes num_layers
        bpc = BitsProbCircuit(num_materialized_nodes, num_layers)
        foreach(pc) do node 
            pid = node2label[node]
            if ismaterializednode(pc, node)
                if isleaf(node)
                    bnode = BitsLiteral(literal(node))
                else
                    child_nodes = children(node)
                    layer = bpc.edge_layers[node2layer[node]-1]
                    if issum(node)
                        bnode = BitsInnerNode(true)
                        for i = 1:length(child_nodes)
                            logp = node.log_probs[i]
                            primeid, subid = prime_sub_ids(pc, child_nodes[i], node2label)
                            fol = first_or_last(i, length(child_nodes))
                            edge = SumEdge(pid, primeid, subid, logp, fol)
                            push!(layer, edge)
                        end
                    else
                        @assert ismul(node)
                        bnode = BitsInnerNode(false)
                        for i = 1:length(child_nodes)
                            primeid, subid = prime_sub_ids(pc, child_nodes[i], node2label)
                            fol = first_or_last(i, length(child_nodes))
                            edge = MulEdge(pid, primeid, subid, fol)
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

    @inline isfirst(x) = (x <= 1)
    @inline islast(x) = (x == 0) || (x == 3)

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

const LocalEdgeProbCircuit = Union{LocalEdgeBitProbCircuits.BitsProbCircuit, LocalEdgeBitProbCircuits.CuProbCircuit}


#########################################################
## Helpers
#########################################################
function bit_node_stats_local(edges::AbstractVector)
    groups = DirectedAcyclicGraphs.groupby(edges) do edge
        if edge isa LocalEdgeBitProbCircuits.SumEdge
            "Sum-$((edge.prime_id > 0) + (edge.sub_id > 0))-$(edge.first_or_last)"
        else
            @assert edge isa LocalEdgeBitProbCircuits.MulEdge
            "Mul-$((edge.prime_id > 0) + (edge.sub_id > 0))-$(edge.first_or_last)"
        end
    end
    DirectedAcyclicGraphs.map_values(v -> length(v), groups, Int)
end

bit_node_stats(bpc::LocalEdgeBitProbCircuits.BitsProbCircuit) =
    bit_node_stats_local(reduce(vcat, bpc.edge_layers))


function balance_threads_local(num_edges, num_examples, config; mine=2, maxe)
    # prefer to assign threads to examples, they do not require memory synchronization
    ex_threads = min(config.threads, num_examples)
    # make sure each thread deals with at most one example
    ex_blocks = cld(num_examples, ex_threads)
    edge_threads = config.threads ÷ ex_threads
    edge_blocks_min = cld(num_edges, edge_threads * maxe)
    edge_blocks_max = cld(num_edges, edge_threads * mine)
    edge_blocks_occupy = cld(config.blocks, ex_blocks)
    # @show edge_blocks_min
    # @show edge_blocks_occupy
    # @show edge_blocks_max
    edge_blocks = min(max(edge_blocks_min, edge_blocks_occupy), edge_blocks_max)
    ((edge_threads, ex_threads), (edge_blocks, ex_blocks))
end

#############################################################
###### Marginals
#############################################################

 

init_mar(node::LocalEdgeBitProbCircuits.BitsInnerNode, data, example_id) = 
    node.issum ? -Inf32 : zero(Float32)

function init_mar(leaf::LocalEdgeBitProbCircuits.BitsLiteral, data, example_id)
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

function init_mar!(mars, bpc::LocalEdgeProbCircuit, data, example_ids)
    broadcast!(mars, bpc.nodes, transpose(example_ids)) do node, example_id
        init_mar(node, data, example_id)
    end
end

function eval_edge!(mars, edge::LocalEdgeBitProbCircuits.SumEdge, example_id)
    child_prob = edge.logp + mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    child_prob
end

function eval_edge!(mars, edge::LocalEdgeBitProbCircuits.MulEdge, example_id)
    child_prob = mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    child_prob
end

function eval_layer!_kernel_local(mars, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    edge_end = min(edge_start + edge_work - 1, length(layer))

    ex_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ex_id <= size(mars,2)
        acc = zero(Float32)    
        local_node = false
        for edge_id = edge_start:edge_end
            edge = layer[edge_id]

            if LocalEdgeBitProbCircuits.isfirst(edge.first_or_last)
                local_node = true
            end

            # compute probability coming from child
            child_prob = mars[edge.prime_id, ex_id]
            if edge.sub_id != 0
                child_prob += mars[edge.sub_id, ex_id]
            end
            if edge isa LocalEdgeBitProbCircuits.SumEdge
                child_prob += edge.logp
            end

            # accumulate probability from child
            if LocalEdgeBitProbCircuits.isfirst(edge.first_or_last) || (edge_id == edge_start)  
                acc = child_prob
            elseif edge isa LocalEdgeBitProbCircuits.SumEdge
                acc = LocalEdgeBitProbCircuits.logsumexp(acc, child_prob)
            else
                acc += child_prob
            end

            # write to global memory
            if LocalEdgeBitProbCircuits.islast(edge.first_or_last) || (edge_id == edge_end)   
                pid = edge.parent_id
                if LocalEdgeBitProbCircuits.islast(edge.first_or_last) && local_node
                    # no one else is writing to this global memory
                    mars[pid, ex_id] = acc
                else
                    if (edge isa LocalEdgeBitProbCircuits.SumEdge)
                        CUDA.@atomic mars[pid, ex_id] = LocalEdgeBitProbCircuits.logsumexp(mars[pid, ex_id], acc)
                    else
                        CUDA.@atomic mars[pid, ex_id] += acc
                    end 
                end             
            end
            
        end
    end
    nothing
end


function eval_layer!(mars, bpc::LocalEdgeProbCircuit, layer_id; mine, maxe, debug=false)
    layer = bpc.edge_layers[layer_id]
    kernel = @cuda name="eval_layer!" launch=false eval_layer!_kernel_local(mars, layer) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads_local(length(layer), size(mars,2), config; mine, maxe)
    debug && println("Layer $layer_id")
    debug && @show config, threads, blocks
    debug && println("Each thread processes $(Float32(length(layer)/threads[1]/blocks[1])) edges")
    if debug
        CUDA.@time kernel(mars, layer; threads, blocks)
    else
        kernel(mars, layer; threads, blocks)
    end
    nothing
end

# run entire circuit
function eval_circuit!(mars, bpc::LocalEdgeProbCircuit, data, example_ids; mine, maxe, debug=false)
    init_mar!(mars, bpc, data, example_ids)
    for i in 1:LocalEdgeBitProbCircuits.num_edge_layers(bpc)
        eval_layer!(mars, bpc, i; mine, maxe, debug)
    end
    nothing
end
