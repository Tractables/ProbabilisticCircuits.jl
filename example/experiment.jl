using Pkg; Pkg.activate("$(@__DIR__)")
using Distributed, ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: BitsProbCircuit

# spawn one worker per device
addprocs(length(devices()))
@everywhere begin
    using Pkg; Pkg.activate("$(@__DIR__)")
    using CUDA
end

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end

@everywhere include("experiment_framework.jl");

@everywhere begin
    println("Loading Data")
    include("load_mnist.jl");

    println("Loading Circuit")
    pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc");
    bpc = BitsProbCircuit(pc);
end

experiments = begin 
    exps = Experiment[]
    batch_size = 512
    epochs_1 = 3#200
    epochs_2 = 3#100
    for pseudocount in [0.1, 0.001]
        begin
            phases = [FullTrainingPhase(epochs_1 + epochs_2, batch_size, pseudocount, NaN)]
            push!(exps, Experiment(phases, nothing))
        end
        for shuffle in [:each_epoch, :each_batch]
            for param_inertia in [0.90, 0.95] #, 0.99, 0.999]
                for flow_memory in [0]
                    phases = [MiniTrainingPhase(
                                epochs_1, 
                                batch_size, 
                                pseudocount,
                                param_inertia,
                                flow_memory,
                                shuffle, NaN),
                            FullTrainingPhase(
                                epochs_2, 
                                batch_size, 
                                pseudocount, NaN)]
                                
                    push!(exps, Experiment(phases, nothing))
                end
            end
            for param_inertia in [0]
                for flow_memory in [60000, 30000] #, 10000]
                    phases = [MiniTrainingPhase(
                                epochs_1, 
                                batch_size, 
                                pseudocount,
                                param_inertia,
                                flow_memory,
                                shuffle, NaN),
                            FullTrainingPhase(
                                epochs_2, 
                                batch_size, 
                                pseudocount, NaN)]
                                
                    push!(exps, Experiment(phases, nothing))
                end
            end
        end
    end
    exps
end


experiments_done = execute(experiments, bpc, train_cpu)

best_experiments = sort(experiments_done, by = e -> e.training_phases[end].last_train_ll)
foreach(e -> println(e), best_experiments)

nothing