using Pkg; Pkg.activate("$(@__DIR__)")
using Revise

includet("experiment_framework.jl");

println("Loading Data")
includet("load_mnist.jl");

println("Loading Circuit")
pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc");
bpc = BitsProbCircuit(pc);

experiments = begin 
    exps = Experiment[]
    batch_size = 512
    epochs_1 = 200
    epochs_2 = 100
    for pseudocount in [0.1, 0.001]
        begin
            phases = [FullTrainingPhase(epochs_1 + epochs_2, batch_size, pseudocount, NaN)]
            push!(exps, Experiment(phases, nothing))
        end
        for shuffle in [:each_epoch, :each_batch]
            for param_inertia in [0.90, 0.95, 0.99, 0.999]
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
                for flow_memory in [10000,30000,60000]
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

execute(experiments, bpc, train_cpu)

best_experiments = sort(experiments, by = e -> e.training_phases[end].last_train_ll)
foreach(e -> println(e), best_experiments)
