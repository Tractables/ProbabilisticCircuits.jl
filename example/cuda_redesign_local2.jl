using Pkg; Pkg.activate(@__DIR__)
using CUDA, LogicCircuits, ProbabilisticCircuits, DataFrames, BenchmarkTools, DirectedAcyclicGraphs
CUDA.allowscalar(false)

pc_file = "meihua_hclt.jpc"
# pc_file = "meihua_hclt_small.jpc"
# pc_file = "rat_mnist_r10_l10_d4_p20.jpc"
# pc_file = "mnist_hclt_cat16.jpc"

# @time pc = read(pc_file, ProbCircuit)
@time pc = ProbabilisticCircuits.read_fast(pc_file)
num_nodes(pc), num_edges(pc)
node_stats(pc)

# generate some fake data
# TODO; figure out row vs col major
data = Array{Union{Bool,Missing}}(replace(rand(0:2, num_variables(pc), 10000), 2 => missing));
data[:,1] .= missing;
cu_data = to_gpu(data);

# create minibatch
batchsize = 512
batch_i = 1:batchsize;
cu_batch_i = CuVector(1:batchsize);

# custom bits circuit

module BitsProbCircuits
        
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
        @show num_materialized_nodes num_layers
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

end

@time bpc, node2label = BitsProbCircuits.BitsProbCircuit(pc);
@time cu_bpc = BitsProbCircuits.CuProbCircuit(bpc);

function bit_node_stats(edges::AbstractVector)
    groups = DirectedAcyclicGraphs.groupby(edges) do edge
        if edge isa BitsProbCircuits.SumEdge
            "Sum-$((edge.prime_id > 0) + (edge.sub_id > 0))-$(edge.first_or_last)"
        else
            @assert edge isa BitsProbCircuits.MulEdge
            "Mul-$((edge.prime_id > 0) + (edge.sub_id > 0))-$(edge.first_or_last)"
        end
    end
    DirectedAcyclicGraphs.map_values(v -> length(v), groups, Int)
end

bit_node_stats(bpc::BitsProbCircuits.BitsProbCircuit) =
    bit_node_stats(reduce(vcat, bpc.edge_layers))


# for i = 1:BitsProbCircuits.num_edge_layers(bpc)
#     println("Layer $i/$(BitsProbCircuits.num_edge_layers(bpc)): $(length(bpc.edge_layers[i])) edges")
# end
bit_node_stats(bpc)
BitsProbCircuits.num_edge_layers(bpc), length(bpc.nodes)

# allocate memory for MAR
mars = Matrix{Float32}(undef, length(bpc.nodes), length(batch_i));
cu_mars = cu(mars);

# custom MAR initialization kernels
init_mar(node::BitsProbCircuits.BitsInnerNode, data, example_id) = 
    node.issum ? -Inf32 : zero(Float32)

function init_mar(leaf::BitsProbCircuits.BitsLiteral, data, example_id)
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

function init_mar!(mars, bpc, data, example_ids)
    broadcast!(mars, bpc.nodes, transpose(example_ids)) do node, example_id
        init_mar(node, data, example_id)
    end
end

# initialize node marginals
# @time init_mar!(mars, bpc, data, batch_i);
CUDA.@time init_mar!(cu_mars, cu_bpc, cu_data, cu_batch_i);

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

function eval_edge!(mars, edge::BitsProbCircuits.SumEdge, example_id)
    child_prob = edge.logp + mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    child_prob
end

function eval_edge!(mars, edge::BitsProbCircuits.MulEdge, example_id)
    child_prob = mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    child_prob
end

isfirst(x) = (x <= 1)
islast(x) = (x == 0) || (x == 3)

function eval_layer!_kernel(mars, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    edge_end = min(edge_start + edge_work - 1, length(layer))

    ex_id = ((blockIdx().y - 1) * blockDim().y + threadIdx().y - 1) + 1

    if ex_id <= size(mars,2)
        acc = Float32(42)    
        local_node = false
        for edge_id = edge_start:edge_end

            edge = layer[edge_id]

            if isfirst(edge.first_or_last)
                local_node = true
            end

            child_prob = mars[edge.prime_id, ex_id]
            if edge.sub_id > 0
                child_prob += mars[edge.sub_id, ex_id]
            end
            if (edge isa BitsProbCircuits.SumEdge)
                child_prob += edge.logp
            end

            if isfirst(edge.first_or_last) || (edge_id == edge_start)  
                acc = child_prob
            elseif edge isa BitsProbCircuits.MulEdge
                acc += child_prob
            else
                acc = logsumexp(acc, child_prob)
            end

            if islast(edge.first_or_last) || (edge_id == edge_end)   
                pid = edge.parent_id
                if islast(edge.first_or_last) && local_node
                    # no one else is writing to this global memory
                    mars[pid, ex_id] = acc
                else
                    if (edge isa BitsProbCircuits.MulEdge)
                        CUDA.@atomic mars[pid, ex_id] += acc
                    else
                        CUDA.@atomic mars[pid, ex_id] = logsumexp(mars[pid, ex_id], acc)
                    end 
                end             
            end
            
        end
    end
    nothing
end

@device_code_warntype @cuda eval_layer!_kernel(cu_mars, cu_bpc.edge_layers[1])

function balance_threads(num_nodes, num_examples, config; block_multiplier)
    # prefer to assign threads to examples, they do not require memory synchronization
    ex_threads = min(config.threads, num_examples)
    # make sure each thread deals with at most one example
    ex_blocks = cld(num_examples, ex_threads)
    node_threads = config.threads ÷ ex_threads
    node_blocks = min(cld(block_multiplier * config.blocks, ex_blocks), cld(num_nodes, node_threads))
    ((node_threads, ex_threads), (node_blocks, ex_blocks))
end

function eval_layer!(mars, bpc, layer_id; block_multiplier, debug=false)
    layer = bpc.edge_layers[layer_id]
    kernel = @cuda name="eval_layer!" launch=false eval_layer!_kernel(mars, layer) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), size(mars,2), config; block_multiplier)
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

# run 1 layer
# @time eval_layer!(mars, bpc, 1);
eval_layer!(cu_mars, cu_bpc, 1; block_multiplier=500, debug=true);

@btime CUDA.@sync eval_layer!(cu_mars, cu_bpc, 1; block_multiplier=500, debug=false);

# run entire circuit
function eval_circuit!(mars, bpc, data, example_ids; block_multiplier, debug=false)
    init_mar!(mars, bpc, data, example_ids)
    for i in 1:BitsProbCircuits.num_edge_layers(bpc)
        eval_layer!(mars, bpc, i; block_multiplier, debug)
    end
    nothing
end

# run all layers
# @time eval_circuit!(mars, bpc, data, batch_i);
CUDA.@time eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i; block_multiplier=500, debug=false);

cu_mars

####################################################
# benchmark node marginals for minibatch
####################################################

# try current MAR code
@btime CUDA.@sync eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i; block_multiplier=500); # new GPU code

# batch_df = to_gpu(DataFrame(transpose(data[:, batch_i]), :auto));
# pbc = to_gpu(ParamBitCircuit(pc, batch_df));
# CUDA.@time reuse = marginal_all(pbc, batch_df);
# @btime CUDA.@sync marginal_all(pbc, batch_df, reuse); # old GPU code

# @btime CUDA.@sync eval_circuit!(mars, bpc, data, batch_i); # new CPU code

