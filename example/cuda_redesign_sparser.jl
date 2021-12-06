using Pkg; Pkg.activate(@__DIR__)
using CUDA, LogicCircuits, ProbabilisticCircuits, DataFrames, BenchmarkTools, DirectedAcyclicGraphs
CUDA.allowscalar(false)

# cd(@__DIR__)
# device!(collect(devices())[2])

# pc_file = "meihua_hclt.jpc"
# pc_file = "meihua_hclt_small.jpc"
# pc_file = "rat_mnist_r10_l10_d4_p20.jpc"
pc_file = "mnist_hclt_cat16.jpc"

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
<<<<<<< HEAD
batchsize = 1024
=======
batchsize = 512
>>>>>>> 3cff079 (cleanup)
batch_i = 1:batchsize;
cu_batch_i = CuVector(1:batchsize);

# try current MAR code
batch_df = to_gpu(DataFrame(transpose(data[:, batch_i]), :auto));
pbc = to_gpu(ParamBitCircuit(pc, batch_df));
CUDA.@time reuse = marginal_all(pbc, batch_df);
CUDA.@time marginal_all(pbc, batch_df, reuse);

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

end

@time bpc, node2label = BitsProbCircuits.BitsProbCircuit(pc);
@time cu_bpc = BitsProbCircuits.CuProbCircuit(bpc);

function bit_node_stats(nodes::AbstractVector)
    groups = DirectedAcyclicGraphs.groupby(nodes) do n
        if n isa BitsProbCircuits.SumEdge
            "Sum-$((n.prime_id_1 > 0) + (n.sub_id_1 > 0))-$((n.prime_id_2 > 0) + (n.sub_id_2 > 0))-$(n.sync)"
        else
            @assert n isa BitsProbCircuits.MulEdge
            "Mul-$((n.prime_id > 0) + (n.sub_id > 0))"
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

function logincexp(matrix::CuDeviceArray, i, j, v, sync)
    if sync
        CUDA.@atomic matrix[i,j] = logsumexp(matrix[i,j], v)
    else
        matrix[i,j] = logsumexp(matrix[i,j], v)
    end
    nothing
end

function logincexp(matrix::Array, i, j, v, sync)
    # should also be atomic for CPU multiprocessing?
    matrix[i,j] = logsumexp(matrix[i,j], v)
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

function eval_edge!(mars, edge::BitsProbCircuits.SumEdge, example_id)
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
        edge_prob = logsumexp(edge_prob, edge_prob_2)
    end
    # increment node value
    logincexp(mars, edge.parent_id, example_id, edge_prob, edge.sync)
    nothing
end

function eval_edge!(mars, edge::BitsProbCircuits.MulEdge, example_id)
    child_prob = mars[edge.prime_id, example_id]
    if edge.sub_id > 0
        child_prob += mars[edge.sub_id, example_id]
    end
    logmulexp(mars, edge.parent_id, example_id, child_prob)
    nothing
end

function eval_layer!(mars::Array, bpc, layer_id)
    for edge in bpc.edge_layers[layer_id]
        for example_id in 1:size(mars,2)
            eval_edge!(mars, edge, example_id)
        end
    end
    nothing
end

function eval_layer!_kernel(mars, layer)
    x_work = ceil(Int, length(layer) / (blockDim().x * gridDim().x))
    x_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * x_work + 1
    x_end = min(x_start + x_work - 1, length(layer))

    y_work = ceil(Int, size(mars,2) / (blockDim().y * gridDim().y))
    y_start = ((blockIdx().y - 1) * blockDim().y + threadIdx().y - 1) * y_work + 1
    y_end = min(y_start + y_work - 1, size(mars,2))

    for edge_id = x_start:x_end
        # for example_id = y_start:y_end
            example_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            eval_edge!(mars, layer[edge_id], y_start)
        # end
    end
    nothing
end

function balance_threads(num_nodes, num_examples, total_threads_per_block)
    ex_threads = min(total_threads_per_block, num_examples)
    ex_blocks = ceil(Int, num_examples / ex_threads)
    node_threads = total_threads_per_block ÷ ex_threads
    node_blocks = ceil(Int, num_nodes / node_threads)
    ((node_threads, ex_threads), (node_blocks, ex_blocks))
end

function eval_layer!(mars, bpc, layer_id)
    layer = bpc.edge_layers[layer_id]
    kernel = @cuda name="eval_layer!" launch=false eval_layer!_kernel(mars, layer) 
    config = launch_configuration(kernel.fun)
    # @show config
    threads, blocks = balance_threads(length(layer), size(mars,2), config.threads)
    # @show threads, blocks
    kernel(mars, layer; threads, blocks)
    nothing
end

# run entire circuit
function eval_circuit!(mars, bpc, data, example_ids)
    init_mar!(mars, bpc, data, example_ids)
    for i in 1:BitsProbCircuits.num_edge_layers(bpc)
        eval_layer!(mars, bpc, i)
    end
    nothing
end

function marginal2(bpc::BitsProbCircuits.BitsProbCircuit, data::CuArray; cu_mars = nothing, batch_size)
    num_examples = size(data)[2]
    cu_ans = CuArray{Float32}(undef, num_examples)
    
    cu_bpc = BitsProbCircuits.CuProbCircuit(bpc); # Around 0.002252 seconds for mnist_hclt_cat16
    if isnothing(cu_mars) || size(cu_mars)[2] != batch_size
        cu_mars = cu(Matrix{Float32}(undef, length(bpc.nodes), batch_size)); # around 0.12 seconds for 4096 bach_size
    end

    for b_ind = 1 : ceil(Int32, num_examples / batch_size)
        batch_start = (b_ind-1) * batch_size + 1
        batch_end   = min(batch_start + batch_size - 1, num_examples)
        
        cu_batch_i = CuArray(batch_start:batch_end)
        if batch_end == num_examples && (batch_end - batch_start + 1 != batch_size)
            # Last batch smaller size
            cur_batch_size = batch_end - batch_start + 1
            init_mar!(cu_mars[:, 1:cur_batch_size], cu_bpc, cu_data, cu_batch_i);
            eval_circuit!(cu_mars[:, 1:cur_batch_size], cu_bpc, cu_data, cu_batch_i);
            cu_ans[cu_batch_i] .= cu_mars[end, 1:cur_batch_size]
        else
            init_mar!(cu_mars, cu_bpc, cu_data, cu_batch_i);
            eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i);
            cu_ans[cu_batch_i] .= cu_mars[end, :]
        end        
        
    end
    return cu_ans
end

####################################################
# benchmark node marginals for minibatch
####################################################

# initialize node marginals
# @time init_mar!(mars, bpc, data, batch_i);
CUDA.@time init_mar!(cu_mars, cu_bpc, cu_data, cu_batch_i);

# run 1 layer
# @time eval_layer!(mars, bpc, 1);
CUDA.@time eval_layer!(cu_mars, cu_bpc, 1);

# run all layers
# @time eval_circuit!(mars, bpc, data, batch_i);
CUDA.@time eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i);

@btime CUDA.@sync marginal_all(pbc, batch_df, reuse); # old GPU code
# @btime CUDA.@sync eval_circuit!(mars, bpc, data, batch_i); # new CPU code
@btime CUDA.@sync eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i); # new GPU code

####################################################
# benchmark marginal likelihood give data set
####################################################

# new gpu w/ preallocated cu_mars
cu_mars2 = cu(Matrix{Float32}(undef, length(bpc.nodes), batchsize));
@btime CUDA.@sync marginal2(bpc, cu_data; cu_mars=cu_mars2, batch_size=batchsize);

# old gpu batched
data_df_batched = to_gpu(batch(DataFrame(transpose(data), :auto), batchsize));
@btime CUDA.@sync marginal_log_likelihood_avg(pbc, data_df_batched);