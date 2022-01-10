using Pkg; Pkg.activate("$(@__DIR__)/../")
using CUDA, LogicCircuits, ProbabilisticCircuits, DataFrames, BenchmarkTools, DirectedAcyclicGraphs
CUDA.allowscalar(false)


module BitsProbCircuits
    
    using DirectedAcyclicGraphs, ProbabilisticCircuits, CUDA

    struct BitsLiteral
        literal::Int
    end

    struct BitsInnerNode
        issum::Bool
    end

    const BitsNode = Union{BitsLiteral,BitsInnerNode}

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
        edge_layers_up::Vector{EdgeLayer}
        edge_layers_down::Vector{EdgeLayer}
    end

    first_or_last(i, n) = 
        (i == n == 1) ? 0 : (i == 1) ? 1 : (i < n) ? 2 : 3

    struct NodeInfo
        prime_id::Int
        sub_id::Int
        layer_id::Int
    end

    function BitsProbCircuit(pc; eager_materialize = true)
        # construct upward edge layers
        nodes, edge_layers_up, par_edges, node2id = BitsProbCircuit_upward(pc; eager_materialize)
        
        # construct downward edge layers
        edge_layers_down = BitsProbCircuit_downward(pc, nodes, par_edges, node2id)
    
        BitsProbCircuit(nodes, edge_layers_up, edge_layers_down)
    end

    function BitsProbCircuit_upward(pc; eager_materialize = true)
        nodes = BitsNode[]
        layers = EdgeLayer[]
    
        # mapping from PC node to its ID
        node2id = Dict{ProbCircuit,UInt32}()
    
        # mapping from node to edges connecting to its parents
        par_edges = Dict{UInt32,Vector{BitsEdge}}()
    
        add_edge(layer_id, edge) = begin
            while length(layers) < layer_id
                push!(layers, EdgeLayer())
            end
            push!(layers[layer_id], edge)
            
            if !haskey(par_edges, edge.prime_id)
                par_edges[edge.prime_id] = BitsEdge[]
            end
            if edge.sub_id != 0 && !haskey(par_edges, edge.sub_id)
                par_edges[edge.sub_id] = BitsEdge[]
            end
            push!(par_edges[edge.prime_id], edge)
            if edge.sub_id != 0
                push!(par_edges[edge.sub_id], edge)
            end
        end
    
        add_node(pc_node, node) = begin
            push!(nodes, node)
            node_id = length(nodes)
            node2id[pc_node] = node_id
            node_id
        end
    
        f_leaf(node) = begin
            node_id = add_node(node, BitsLiteral(literal(node)))
            layer_id = 0
            NodeInfo(node_id, 0, layer_id)
        end
    
        f_inner(node, children_info) = begin
            if (length(children_info) == 1 && (!eager_materialize || children_info[1].sub_id == 0))
                # this is a pass-through node
                children_info[1]
            elseif (ismul(node) && length(children_info) == 2 
                    && children_info[1].sub_id == 0 && children_info[2].sub_id == 0)
                # this is a simple conjunctive element that we collapse into an edge
                layer_id = max(children_info[1].layer_id, children_info[2].layer_id)
                NodeInfo(children_info[1].prime_id, children_info[2].prime_id, layer_id)
            else
                # materialize the current node
                node_id = add_node(node, BitsInnerNode(issum(node)))
                layer_id = maximum(x -> x.layer_id, children_info) + 1
                if issum(node)
                    ch_len = length(children_info)
                    for i = 1 : ch_len
                        logp = node.log_probs[i]
                        child_info = children_info[i]
                        fol = first_or_last(i, ch_len)
                        edge = SumEdge(node_id, child_info.prime_id, child_info.sub_id, logp, fol)
                        add_edge(layer_id, edge)
                    end
                else
                    @assert ismul(node)
                    single_infos = filter(x -> (x.sub_id == 0), children_info)
                    double_infos = filter(x -> (x.sub_id != 0), children_info)
                    sin_ch_len = length(single_infos)
                    for i = 1 : 2 : sin_ch_len
                        if i < sin_ch_len
                            merged_layer_id = max(single_infos[i].layer_id, single_infos[i+1].layer_id)
                            merged_info = NodeInfo(single_infos[i].prime_id, single_infos[i+1].prime_id, merged_layer_id)
                            single_infos[i] = merged_info
                        end
                        push!(double_infos, single_infos[i])
                    end
                    dou_ch_len = length(double_infos)
                    for i = 1 : dou_ch_len
                        child_info = double_infos[i]
                        fol = first_or_last(i, dou_ch_len)
                        edge = MulEdge(node_id, child_info.prime_id, child_info.sub_id, fol)
                        add_edge(layer_id, edge)
                    end
                end
                NodeInfo(node_id, 0, layer_id)
            end
        end
    
        root_info = foldup_aggregate(pc, f_leaf, f_inner, NodeInfo)
    
        if root_info.sub_id != 0
            # manually materialize root node
            @assert ismul(pc)
            @assert num_children(pc) == 1
            node_id = add_node(node, BitsInnerNode(false))
            edge = MulEdge(node_id, root_info.prime_id, root_info.sub_id, first_or_last(1, 1))
            add_edge(root_info.layer_id + 1, edge)
        end
    
        nodes, layers, par_edges, node2id
    end

    function BitsProbCircuit_downward(pc, nodes, par_edges, node2id)
        layers = EdgeLayer[]
    
        node2layer_id = Dict{UInt32,UInt32}()
        node2layer_id[node2id[pc]] = zero(UInt32) # The root node is at layer 0
    
        added_edges = Set{Tuple{UInt32,UInt32,UInt32}}()
    
        add_edge(layer_id, edge) = begin
            while length(layers) < layer_id
                push!(layers, EdgeLayer())
            end
            push!(layers[layer_id], edge)
        end
    
        get_layer_id(node_id) = 
            maximum(map(e->node2layer_id[e.parent_id], par_edges[node_id])) + 1
    
        construct_edge(old_edge::SumEdge, prime_id, sub_id; status) = begin
            SumEdge(old_edge.parent_id, prime_id, sub_id, old_edge.logp, status)
        end
        construct_edge(old_edge::MulEdge, prime_id, sub_id; status) = begin
            MulEdge(old_edge.parent_id, prime_id, sub_id, status)
        end
        construct_edge(old_edge::SumEdge; status) = begin
            SumEdge(old_edge.parent_id, old_edge.prime_id, old_edge.sub_id, old_edge.logp, status)
        end
        construct_edge(old_edge::MulEdge; status) = begin
            MulEdge(old_edge.parent_id, old_edge.prime_id, old_edge.sub_id, status)
        end
    
        foreach_down(pc) do pn
            if haskey(node2id, pn) && pn !== pc
                node_id = node2id[pn]
                layer_id = get_layer_id(node_id)
                npar = length(par_edges[node_id])
                @assert npar >= 1
            
                node2layer_id[node_id] = layer_id
            
                has_updated_edge = false
            
                new_edges = map(par_edges[node_id]) do edge
                    sub_id = (edge.prime_id == node_id) ? edge.sub_id : edge.prime_id
                    sub_npar = (sub_id == 0) ? 1 : length(par_edges[sub_id])
                    if (edge.parent_id, node_id, sub_id) in added_edges
                        has_updated_edge = true
                        nothing
                    elseif npar == 1 && sub_npar == 1
                        if edge.prime_id == node_id # only record this edge once
                            construct_edge(edge, node_id, sub_id; status = 0)
                        else
                            nothing
                        end
                    elseif sub_npar == 1 # npar > 1
                        construct_edge(edge, node_id, sub_id; status = 0)
                    elseif npar > 1 && sub_npar > 1
                        construct_edge(edge, node_id, sub_id; status = 4)
                    else
                        nothing
                    end
                end
                new_edges = filter(e->(e !== nothing), new_edges)
            
                for new_edge in new_edges
                    push!(added_edges, (new_edge.parent_id, new_edge.prime_id, new_edge.sub_id))
                    push!(added_edges, (new_edge.parent_id, new_edge.sub_id, new_edge.prime_id))
                end
            
                if !has_updated_edge
                    if length(new_edges) > 1
                        for (i, new_edge) in enumerate(new_edges)
                            if i == 1
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 1))
                            elseif i == length(new_edges)
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 3))
                            else
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 2))
                            end
                        end
                    elseif length(new_edges) == 1
                        add_edge(layer_id, new_edges[1])
                    end
                else
                    if length(new_edges) > 1
                        for (i, new_edge) in enumerate(new_edges)
                            if i == 1
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 2))
                            elseif i == length(new_edges)
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 3))
                            else
                                add_edge(layer_id, construct_edge(new_edge; status = new_edge.first_or_last + 2))
                            end
                        end
                    elseif length(new_edges) == 1
                        add_edge(layer_id, construct_edge(new_edges[1]; status = new_edge.first_or_last + 2))
                    end
                end
            end
        end
    
        layers
    end

    num_edge_layers_up(bpc::AbstractBitsProbCircuit) = 
        length(bpc.edge_layers_up)
    num_edge_layers_down(bpc::AbstractBitsProbCircuit) = 
        length(bpc.edge_layers_down)

    const CuEdgeLayer = CuVector{BitsEdge}

    struct CuProbCircuit <: AbstractBitsProbCircuit
        nodes::CuVector{BitsNode}
        edge_layers_up::Vector{CuEdgeLayer}
        edge_layers_down::Vector{CuEdgeLayer}
        CuProbCircuit(bpc::BitsProbCircuit) = begin
            nodes = cu(bpc.nodes)
            edge_layers_up = map(cu, bpc.edge_layers_up)
            edge_layers_down = map(cu, bpc.edge_layers_down)
            new(nodes, edge_layers_up, edge_layers_down)
        end
    end

end

@inline isfirst(x) = (x <= 1)
@inline islast(x) = (x == 0) || (x == 3)

function balance_threads(num_edges, num_examples, config; mine=2, maxe)
    # prefer to assign threads to examples, they do not require memory synchronization
    ex_threads = min(config.threads, num_examples)
    ex_blocks = cld(num_examples, ex_threads)
    edge_threads = config.threads ÷ ex_threads
    edge_blocks_min = cld(num_edges, edge_threads * maxe)
    edge_blocks_max = cld(num_edges, edge_threads * mine)
    edge_blocks_occupy = cld(config.blocks, ex_blocks)
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


function eval_layer!(mars, bpc, layer_id; mine, maxe, debug=false)
    layer = bpc.edge_layers_up[layer_id]
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


function eval_circuit!(mars, bpc, data, example_ids; mine, maxe, debug=false)
    init_mar!(mars, bpc, data, example_ids; mine, maxe, debug)
    for i in 1:BitsProbCircuits.num_edge_layers_up(bpc)
        eval_layer!(mars, bpc, i; mine, maxe, debug)
    end
    nothing
end


@inline isfirst_down(x) = isfirst(x % 4)
@inline islast_down(x) = islast(x % 4)
@inline sub_is_fandl(x) = (x < 4)

function init_flow!(flows)
    @inbounds @views flows[:,:] .= typemin(eltype(flows))
    @inbounds @views flows[:,end] .= zero(eltype(flows))
end


function eval_layer_down!_kernel(mars, flows, layer)
    edge_work = cld(length(layer), (blockDim().x * gridDim().x))
    edge_start = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) * edge_work + 1
    edge_end = min(edge_start + edge_work - 1, length(layer))
    
    ex_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if ex_id <= size(mars, 1)
        acc = typemin(Float32)
        local_node = false
        for edge_id = edge_start : edge_end
            edge = layer[edge_id]
            
            if isfirst_down(edge.first_or_last)
                local_node = true
            end
            
            # compute edge flow
            if edge isa BitsProbCircuits.SumEdge
                edge_flow = mars[ex_id, edge.prime_id] - mars[ex_id, edge.parent_id] + flows[ex_id, edge.parent_id] + edge.logp
                if edge.sub_id != 0
                    edge_flow += mars[ex_id, edge.sub_id]
                end
            else # edge isa BitsProbCircuits.MulEdge
                edge_flow = flows[ex_id, edge.parent_id]
            end
            
            # accumulate edge flow to node flow
            if isfirst_down(edge.first_or_last) || (edge_id == edge_start)
                acc = edge_flow
            else
                acc = logsumexp(acc, edge_flow)
            end
            
            # write to global memory
            if islast_down(edge.first_or_last) || (edge_id == edge_end)
                prime_id = edge.prime_id
                if islast_down(edge.first_or_last) && local_node
                    # no one else is writing to this global memory
                    flows[ex_id, prime_id] = acc
                    local_node = false
                else
                    CUDA.@atomic flows[ex_id, prime_id] = logsumexp(flows[ex_id, prime_id], acc)
                end
            end
            if edge.sub_id != 0
                if sub_is_fandl(edge.first_or_last)
                    flows[ex_id, edge.sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, edge.sub_id] = logsumexp(flows[ex_id, edge.sub_id], edge_flow)
                end
            end
        end
    end
    
    nothing
end


function eval_layer_down!(mars, flows, bpc, layer_id; mine, maxe, debug=false)
    layer = bpc.edge_layers_down[layer_id]
    kernel = @cuda name="eval_layer_down!" launch=false eval_layer_down!_kernel(mars, flows, layer) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), size(mars,1), config; mine, maxe)
    debug && println("Layer $layer_id")
    debug && @show config, threads, blocks
    debug && println("Each thread processes $(Float32(length(layer)/threads[1]/blocks[1])) edges")
    if debug
        CUDA.@time kernel(mars, flows, layer; threads, blocks)
    else
        kernel(mars, flows, layer; threads, blocks)
    end
    nothing
end


function eval_circuit_down!(mars, flows, bpc; mine, maxe, debug=false)
    init_flow!(flows)
    for i in BitsProbCircuits.num_edge_layers_down(bpc) : -1 : 1
        eval_layer_down!(mars, flows, bpc, i; mine, maxe, debug)
    end
    nothing
end

##################################################################################
##################################################################################

pc_file = "meihua_hclt.jpc"
# pc_file = "meihua_hclt_small.jpc"
# pc_file = "rat_mnist_r10_l10_d4_p20.jpc"
# pc_file = "mnist_hclt_cat16.jpc"
@time pc = ProbabilisticCircuits.read_fast(pc_file)

# pc_file = "ad_12_16.jpc.gz"
# @time pc = read(pc_file, ProbCircuit)

# @time pc = zoo_psdd("plants.psdd")

@time bpc = BitsProbCircuits.BitsProbCircuit(pc);
@time cu_bpc = BitsProbCircuits.CuProbCircuit(bpc);

# allocate memory for MAR and FLOW
mars = Matrix{Float32}(undef, length(batch_i), length(bpc.nodes));
cu_mars = cu(mars);
flows = Matrix{Float32}(undef, length(batch_i), length(bpc.nodes));
cu_flows = cu(flows);

@benchmark CUDA.@sync eval_circuit!(cu_mars, cu_bpc, cu_data, cu_batch_i; mine=2, maxe=16, debug=false)
@benchmark CUDA.@sync eval_circuit_down!(cu_mars, cu_flows, cu_bpc; mine=2, maxe=16, debug=false)

##################################################################################
##################################################################################

# try current MAR+flow code as baseline
batch_df = to_gpu(DataFrame(transpose(data[:, batch_i]), :auto));
pbc = to_gpu(ParamBitCircuit(pc, batch_df));
reuse = marginal_all(pbc, batch_df);
reuse2 = marginal_flows_down(pbc, reuse);

@benchmark CUDA.@sync marginal_all(pbc, batch_df, reuse)
@benchmark CUDA.@sync marginal_flows_down(pbc, reuse, reuse2)
