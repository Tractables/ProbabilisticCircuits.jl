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
end

function setup_gpu_memory(bpc, batch_size)
    node_aggr_mem = CuVector{Float32}(undef, length(bpc.nodes));
    edge_aggr_mem = CuVector{Float32}(undef, length(bpc.edge_layers_down.vectors));

    mars_mem = CuMatrix{Float32}(undef, batch_size, length(bpc.nodes));
    flows_mem = similar(mars_mem);

    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem
end

function execute(phase::FullTrainingPhase, bpc, train_data)

    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem = 
        setup_gpu_memory(bpc, phase.batch_size)

    full_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, pseudocount = phase.pseudocount,
        mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem)

    CUDA.unsafe_free!.([node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem])

    nothing
end

function execute(phase::MiniTrainingPhase, bpc, train_data)

    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem = 
        setup_gpu_memory(bpc, phase.batch_size)

    mini_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, pseudocount = phase.pseudocount, 
        param_inertia = phase.param_inertia, flow_memory = phase.flow_memory,
        shuffle = phase.shuffle,
        mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem)

    CUDA.unsafe_free!.([node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem])

    nothing
end

function execute(exper::Experiment, bpc, train_data; logfile)  
    println("Starting experiment: $exper")
    for phase in exper.training_phases
        println("Starting phase: $phase")
        execute(phase, bpc, train_data)
        phase.last_train_ll = 
            loglikelihood(train_data, bpc; batch_size = phase.batch_size)
        println("Done with phase: $phase")
    end
    open(logfile, "a") do io
        println(io, exper)
    end;
    println("Done with experiment: $exper")
    exper
end

function execute(experiments::Vector{Experiment}, bpc, train_data; 
    logfile = "experiments.log")
    println("Starting $(length(experiments)) experiments")  
    open(logfile, "a") do io
        println(io, "=== Starting $(length(experiments)) experiments ===")
    end;
    done_experiments = pmap(experiments) do exper
        execute(exper, bpc(), train_data(); logfile) 
    end
    open(logfile, "a") do io
        println(io, "=== Done with $(length(experiments)) experiments ===")
    end;
    println("Done with $(length(experiments)) experiments")
    done_experiments
end