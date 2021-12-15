using Pkg; Pkg.activate(@__DIR__)
using CUDA, LogicCircuits, ProbabilisticCircuits, DataFrames, BenchmarkTools, DirectedAcyclicGraphs
CUDA.allowscalar(false)

# pc_file = "meihua_hclt.jpc"
# pc_file = "meihua_hclt_small.jpc"
# pc_file = "rat_mnist_r10_l10_d4_p20.jpc"
# pc_file = "mnist_hclt_cat16.jpc"
# @time pc = ProbabilisticCircuits.read_fast(pc_file)

pc_file = "ad_12_16.jpc.gz"
@time pc = read(pc_file, ProbCircuit)

# @time pc = zoo_psdd("plants.psdd")

num_nodes(pc), num_edges(pc)
node_stats(pc)

# generate some fake data
# TODO; figure out row vs col major
data = Array{Union{Bool,Missing}}(replace(rand(0:2, 10000, num_variables(pc)), 2 => missing));
data[1,:] .= missing;
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
mars = Matrix{Float32}(undef, length(batch_i), length(bpc.nodes));
cu_mars = cu(mars);

function balance_threads(num_edges, num_examples, config; mine=2, maxe)
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

function init_mar!_kernel(mars, nodes, data, example_ids)
    node_work = cld(length(nodes), (blockDim().x * gridDim().x))
    node_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * node_work + 1
    node_end = min(node_start + node_work - 1, length(nodes))

    ex_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ex_id <= size(mars,1)
        for node_id = node_start:node_end
            node = nodes[node_id]
            mars[ex_id, node_id] = 
                if (node isa BitsProbCircuits.BitsInnerNode)
                    node.issum ? -Inf32 : zero(Float32)
                else
                    orig_ex_id = example_ids[ex_id]
                    leaf = node::BitsProbCircuits.BitsLiteral
                    lit = leaf.literal
                    v = data[orig_ex_id, abs(lit)]
                    if ismissing(v)
                        zero(Float32)
                    elseif (lit > 0) == v
                        zero(Float32)
                    else
                        -Inf32
                    end
                end
        end
    end
    nothing
end

# @device_code_warntype @cuda init_mar!_kernel(cu_mars, cu_bpc.nodes, cu_data, cu_batch_i)

function init_mar!(mars, bpc, data, example_ids; mine, maxe, debug=false)
    @assert size(mars,1) == length(example_ids)
    kernel = @cuda name="init_mar!" launch=false init_mar!_kernel(mars, bpc.nodes, data,example_ids) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(bpc.nodes), length(example_ids), config; mine, maxe)
    debug && println("Node initialization")
    debug && @show config, threads, blocks
    debug && println("Each thread processes $(Float32(length(bpc.nodes)/threads[1]/blocks[1])) nodes")
    if debug
        CUDA.@time kernel(mars, bpc.nodes, data,example_ids; threads, blocks)
    else
        kernel(mars, bpc.nodes, data,example_ids; threads, blocks)
    end
    nothing
end

# # initialize node marginals
# init_mar!(cu_mars, cu_bpc, cu_data, cu_batch_i; mine=2, maxe=8, debug=true);

# @btime CUDA.@sync init_mar!(cu_mars, cu_bpc, cu_data, cu_batch_i; mine=2, maxe=16, debug=false);

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

@inline isfirst(x) = (x <= 1)
@inline islast(x) = (x == 0) || (x == 3)

function eval_layer!_kernel(mars, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    edge_end = min(edge_start + edge_work - 1, length(layer))

    ex_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ex_id <= size(mars,1)
        acc = zero(Float32)    
        local_node = false
        for edge_id = edge_start:edge_end
            edge = layer[edge_id]

            if isfirst(edge.first_or_last)
                local_node = true
            end

            # compute probability coming from child
            child_prob = mars[ex_id, edge.prime_id]
            if edge.sub_id != 0
                child_prob += mars[ex_id, edge.sub_id]
            end
            if edge isa BitsProbCircuits.SumEdge
                child_prob += edge.logp
            end

            # accumulate probability from child
            if isfirst(edge.first_or_last) || (edge_id == edge_start)  
                acc = child_prob
            elseif edge isa BitsProbCircuits.SumEdge
                acc = logsumexp(acc, child_prob)
            else
                acc += child_prob
            end

            # write to global memory
            if islast(edge.first_or_last) || (edge_id == edge_end)   
                pid = edge.parent_id
                if islast(edge.first_or_last) && local_node
                    # no one else is writing to this global memory
                    mars[ex_id, pid] = acc
                else
                    if (edge isa BitsProbCircuits.SumEdge)
                        CUDA.@atomic mars[ex_id, pid] = logsumexp(mars[ex_id, pid], acc)
                    else
                        CUDA.@atomic mars[ex_id, pid] += acc
                    end 
                end             
            end
            
        end
    end
    nothing
end

# @device_code_warntype @cuda eval_layer!_kernel(cu_mars, cu_bpc.edge_layers[1])

function eval_layer!(mars, bpc, layer_id; mine, maxe, debug=false)
    layer = bpc.edge_layers[layer_id]
    kernel = @cuda name="eval_layer!" launch=false eval_layer!_kernel(mars, layer) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), size(mars,1), config; mine, maxe)
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
eval_layer!(cu_mars, cu_bpc, 1; mine=8, maxe=32, debug=true);

# @btime CUDA.@sync eval_layer!(cu_mars, cu_bpc, 1; block_multiplier=500, debug=false);

# run entire circuit
function eval_circuit!(mars, bpc, data, example_ids; mine, maxe, debug=false)
    init_mar!(mars, bpc, data, example_ids; mine, maxe, debug)
    for i in 1:BitsProbCircuits.num_edge_layers(bpc)
        eval_layer!(mars, bpc, i; mine, maxe, debug)
    end
    nothing
end

# run all layers
# @time eval_circuit!(mars, bpc, data, batch_i);
CUDA.@time eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i; mine=2, maxe=16, debug=true);

####################################################
# benchmark node marginals for minibatch
####################################################

# try current MAR code
function tune() 
    for i=0:7
        for j=i:7
            minev = 2^i
            maxev = 2^j
            println("mine=$minev, maxe=$maxev")
            @btime CUDA.@sync eval_circuit!($cu_mars, $cu_bpc, $cu_data, $cu_batch_i; mine=$minev, maxe=$maxev); # new GPU code
        end
    end
end
tune()
# MNIST 512 mine=2, maxe=8
# RAT-SPN 512 mine=2, maxe=8
# BIO-HCLT32 512 mine=2, maxe=32

# TODO try transposing the MAR matrix

@btime CUDA.@sync eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i; mine=2, maxe=16);


# OLD GPU CODE BENCHMARK
batch_df = to_gpu(DataFrame(data[batch_i,:], :auto));
pbc = to_gpu(ParamBitCircuit(pc, batch_df));
CUDA.@time reuse = marginal_all(pbc, batch_df);
@btime CUDA.@sync marginal_all(pbc, batch_df, reuse); # old GPU code

nothing