using Distributed 

using ProbabilisticCircuits, CUDA, BenchmarkTools
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, probs_flows_circuit, eval_circuit, loglikelihood, flows_circuit, aggr_node_flows, update_params, full_batch_em_step, full_batch_em, mini_batch_em, soften
mutable struct StructLearningPhase
    cats
    num_params
    StructLearningPhase(c) = new(c, NaN)
end

abstract type ParamTrainingPhase end
mutable struct FullTrainingPhase <: ParamTrainingPhase
    epochs
    batch_size
    pseudocount
    softness
    last_train_ll
end

mutable struct MiniTrainingPhase <: ParamTrainingPhase
    epochs
    batch_size
    pseudocount
    softness
    param_inertia
    param_inertia_end
    flow_memory
    flow_memory_end
    shuffle
    last_train_ll
end

mutable struct Experiment
    struct_learning::StructLearningPhase
    training_phases::Vector{ParamTrainingPhase}
    test_ll
    time
    Experiment(sl, phases) = new(sl, phases, NaN, NaN)
end

function execute(phase::StructLearningPhase, train_data)
    circuit = hclt(train_data; num_cats=2, num_hidden_cats = phase.cats)
    phase.num_params = num_parameters(circuit)
    uniform_parameters!(circuit; perturbation = 0.4)
    CuBitsProbCircuit(BitsProbCircuit(circuit))
end

function execute(phase::FullTrainingPhase, bpc, train_data)

    full_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, 
        pseudocount = phase.pseudocount, softness = phase.softness)

    nothing
end

function execute(phase::MiniTrainingPhase, bpc, train_data)

    mini_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, 
        pseudocount = phase.pseudocount, softness = phase.softness, 
        param_inertia = phase.param_inertia, param_inertia_end = phase.param_inertia_end, 
        flow_memory = phase.flow_memory, flow_memory_end = phase.flow_memory_end,
        shuffle = phase.shuffle)

    nothing
end

function execute(exper::Experiment, train_data, test_data; logfile)  
    println("Starting experiment: $exper")
    println("Starting phase: $(exper.struct_learning)")
    bpc = execute(exper.struct_learning, train_data)
    println("Done with phase: $(exper.struct_learning)")
    exper.time = @elapsed for phase in exper.training_phases
        println("Starting phase: $phase")
        execute(phase, bpc, train_data)
        phase.last_train_ll = 
            loglikelihood(train_data, bpc; batch_size = phase.batch_size)
        println("Done with phase: $phase")
    end
    exper.test_ll =
        loglikelihood(test_data, bpc; batch_size = exper.training_phases[end].batch_size)
    open(logfile, "a") do io
        println(io, exper)
    end;
    println("Done with experiment: $exper")
    exper
end

function execute(experiments::Vector{Experiment}, train_data, test_data; 
    logfile = "experiments.log")
    println("Starting $(length(experiments)) experiments")  
    open(logfile, "a") do io
        println(io, "=== Starting $(length(experiments)) experiments ===")
    end;
    done_experiments = pmap(experiments) do exper
        execute(exper, train_data(), test_data(); logfile) 
    end
    open(logfile, "a") do io
        println(io, "=== Done with $(length(experiments)) experiments ===")
    end;
    println("Done with $(length(experiments)) experiments")
    done_experiments
end