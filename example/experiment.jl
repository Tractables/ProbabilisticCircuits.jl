using Pkg; Pkg.activate("$(@__DIR__)")

include("gpu_workers.jl");

@everywhere include("experiment_framework.jl");

experiments = begin 
    exps = Experiment[]
    batch_size = 512
    epochs_1 = 300
    epochs_2 = 100
    for pseudocount in [0.01, 0.001, 0.0001, 0.00001]
        for softness in [0.01, 0.001, 0.0001, 0]
            for shuffle in [:each_batch] #:each_epoch
                push!(exps, Experiment([
                    MiniTrainingPhase(
                        epochs_1, batch_size, 
                        pseudocount, softness,
                        0, 1,
                        0, 0,
                        shuffle, NaN),
                    FullTrainingPhase(
                        epochs_2, batch_size, 
                        pseudocount, softness, NaN)]))
            end
        end
    end
    exps
end;

# main process does no work
train_data() = nothing
test_data() = nothing
pc_model() = nothing

# workers do work
@everywhere workers() begin
    println("Loading Data")
    include("load_mnist.jl");
    train_gpu, test_gpu = mnist_gpu()
    train_data() = train_gpu
    test_data() = test_gpu

    println("Loading Circuit")
    using ProbabilisticCircuits: BitsProbCircuit
    pc = ProbabilisticCircuits.read_fast("mnist_bits_hclt_32.jpc");
    bpc = BitsProbCircuit(pc)
    # make sure to get a fresh circuit each time, to reset the parameters
    pc_model() = CuBitsProbCircuit(bpc)
end

experiments_done = execute(experiments, pc_model, train_data, test_data; logfile = "experiments2.log")

best_experiments = sort(experiments_done, by = e -> e.training_phases[end].last_train_ll)
foreach(e -> println(e), best_experiments)

nothing