using ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em

abstract type TrainingPhase end

mutable struct FullTrainingPhase <: TrainingPhase
    epochs
    batch_size
    pseudocount
    last_train_ll
end

mutable struct MiniTrainingPhase <: TrainingPhase
    epochs
    batch_size
    pseudocount
    param_inertia
    flow_memory
    shuffle
    last_train_ll
end

mutable struct Experiment
    training_phases::Vector{TrainingPhase}
    bpc
end

Base.show(io::IO, ex::Experiment) =
    Base.show(io, ex.training_phases) # hide bpc

function setup_gpu_memory(bpc, batch_size)
    node_aggr_mem = CuVector{Float32}(undef, length(bpc.nodes));
    edge_aggr_mem = CuVector{Float32}(undef, length(bpc.edge_layers_down.vectors));

    mars_mem = CuMatrix{Float32}(undef, batch_size, length(bpc.nodes));
    flows_mem = similar(mars_mem);

    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem
end

function execute(bpc, phase::FullTrainingPhase, train_data::CuArray)
    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem = 
        setup_gpu_memory(bpc, phase.batch_size)

    full_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, pseudocount = phase.pseudocount,
        mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem)
    nothing
end

function execute(bpc, phase::MiniTrainingPhase, train_data::CuArray)
    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem = 
        setup_gpu_memory(bpc, phase.batch_size)

    mini_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, pseudocount = phase.pseudocount, 
        param_inertia = phase.param_inertia, flow_memory = phase.flow_memory,
        shuffle = phase.shuffle,
        mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem)
    nothing
end

function execute(exper::Experiment, bpc::BitsProbCircuit, train_data::Array, logfile)
    tid = Threads.threadid()   
    gpus = collect(devices())

    did = mod1(tid, length(gpus))
    device!(gpus[did])
    println("Starting experiment on thread $tid and GPU $did: $exper")

    exper.bpc = CuBitsProbCircuit(bpc);
    cu_train = cu(train_data)

    for phase in exper.training_phases
        println("Starting phase on thread $tid and GPU $did: $phase")
        execute(exper.bpc, phase, cu_train)
        phase.last_train_ll = 
            loglikelihood(cu_train, exper.bpc; batch_size = phase.batch_size)
        println("Done with phase on thread $tid and GPU $did: $phase")
    end
    open(logfile, "a") do io
        println(io, exper)
    end;
    println("Done with experiment on thread $tid and GPU $did: $exper")
    nothing
end

function execute(experiments::Vector{Experiment}, bpc::BitsProbCircuit, train_data::Array, logfile = "experiments.log")
    println("Starting $(length(experiments)) experiments")
    open(logfile, "a") do io
        println(io, "===Starting $(length(experiments)) experiments===")
    end;
    Threads.@threads for exper in experiments
        execute(exper, bpc, train_data, logfile) 
    end
    open(logfile, "a") do io
        println(io, "===Done with $(length(experiments)) experiments===")
    end;
    println("Done with $(length(experiments)) experiments")
    nothing
end
