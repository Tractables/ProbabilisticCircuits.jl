using Distributed 

num = 2
addprocs(num)

@everywhere begin
    struct testParams{F}
        f::F
        args::Tuple
        kwargs::Pairs
        testParams(f, args...; kwargs...) = new{typeof(f)}(f, args, kwargs)
    end        

    function worker(input, output)
        while true
            p = take!(input)
            put!(output, (myid(), p.f(1.0, p.args...)))
        end
    end
end

# channels to send and receive data from workers
input = RemoteChannel(()->Channel{testParams}(num))
output = RemoteChannel(()->Channel{Tuple}(num))

for i in 1:num
    errormonitor(@async put!(input, testParams(*, 2)))
end

# run workers
for w in workers()
    remote_do(worker, w, input, output)
end

# collect results
for i in 1:num
    @async begin 
        w, ans = take!(output)
        println("worker $w: $ans")
    end
end






mutable struct Experiment
    training_phases::Vector{TrainingPhase}
    test_ll
    time
    Experiment(phases) = new(phases, NaN, NaN)
end

function execute(phase::FullTrainingPhase, bpc, train_data)

    full_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, 
        pseudocount = phase.pseudocount, softness = phase.softness)

    nothing
end

function execute(phase::MiniTrainingPhase, bpc, train_data)

    node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem = 
        setup_gpu_memory(bpc, phase.batch_size)

    mini_batch_em(bpc, train_data, phase.epochs; 
        batch_size = phase.batch_size, 
        pseudocount = phase.pseudocount, softness = phase.softness, 
        param_inertia = phase.param_inertia, param_inertia_end = phase.param_inertia_end, 
        flow_memory = phase.flow_memory, flow_memory_end = phase.flow_memory_end,
        shuffle = phase.shuffle,
        mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem)

    CUDA.unsafe_free!.([node_aggr_mem, edge_aggr_mem, mars_mem, flows_mem])

    nothing
end

function execute(exper::Experiment, bpc, train_data, test_data; logfile)  
    println("Starting experiment: $exper")
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

function execute(experiments::Vector{Experiment}, bpc, train_data, test_data; 
    logfile = "experiments.log")
    println("Starting $(length(experiments)) experiments")  
    open(logfile, "a") do io
        println(io, "=== Starting $(length(experiments)) experiments ===")
    end;
    done_experiments = pmap(experiments) do exper
        execute(exper, bpc(), train_data(), test_data(); logfile) 
    end
    open(logfile, "a") do io
        println(io, "=== Done with $(length(experiments)) experiments ===")
    end;
    println("Done with $(length(experiments)) experiments")
    done_experiments
end