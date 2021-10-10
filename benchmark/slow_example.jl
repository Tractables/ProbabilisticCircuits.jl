using Pkg; Pkg.activate(@__DIR__)

using Revise
using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: MMAPCache, update_and_log
using DataStructures: DefaultDict, counter
using BenchmarkTools
using Profile
using Random
using DataFrames
using JSON
using Serialization
using CSV

Random.seed!(7351);
pc = read("$(@__DIR__)/plants.psdd", ProbCircuit);
myquer = open(deserialize, "$(@__DIR__)/quer.jls");
csv = joinpath("$(@__DIR__)/progress.csv");
log_func(results) = begin
    table = DataFrame(;[Symbol(x) => results[x] for x in MMAP_LOG_HEADER]...)
    CSV.write(csv, table; )
end

function my_mmap_solve(root, quer; num_iter=length(quer), prune_attempts=10, log_per_iter, heur="maxP", out)

    # initialize log
    num_iter = min(num_iter, length(quer))
    results = Dict()
    for x in MMAP_LOG_HEADER
        results[x] = Vector{Union{Any,Missing}}(missing,num_iter)
    end

    # marginalize out non-query variables
    # cur_root = root
    ub = forward_bounds(root, quer)
    _, mp = max_sum_lower_bound(root, quer)
    # @show ub, log(mp)
    cur_root = marginalize_out(root, setdiff(variables(root), quer))
    @assert variables(cur_root) == quer
    # TODO: check the bounds before and after

    splittable = copy(quer)
    cache = MMAPCache()
    ub = forward_bounds(cur_root, quer, cache)
    _, mp = max_sum_lower_bound(cur_root, quer, cache)
    lb = log(mp)
    # @show ub, lb
    counters = counter(Int)
    for i in 1:num_iter
        if i == 1
            # println("$i - A")
            if isempty(splittable) || ub < lb + 1e-10
                break
            end

            # Prune -- could improve the upper bound (TODO: maybe also the lower bound?)
            # println(out, "* Starting with $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            tic = time_ns()
            # cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, exp(lb), prune_attempts)
            cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, lb, prune_attempts)
            merge!(counters, prune_counters)
            toc = time_ns()
            # println(out, "* Pruning gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            # update_and_log(cur_root,root,quer,cache,lb,results,i,true, prune_attempts=actual_reps, time=(toc-tic)/1.0e9)

            # Split root (move a quer variable up) -- could improve both the upper and lower bounds
            tic = time_ns()
            to_split = get_to_split(cur_root, splittable, counters, heur, lb)
            cur_root = add_and_split(cur_root, to_split)
            toc = time_ns()
            # println(out, "* Splitting on $(to_split) gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            delete!(splittable, to_split)
            # ub, lb = update_and_log(cur_root,root,quer,cache,lb,results,i,false, split_var=to_split, time=(toc-tic)/1.0e9)

            log_per_iter(results)
            # TODO: also save the circuit at the end of each iteration for easy retrieval?    
        else
            
            @profile @time begin
                
                if isempty(splittable) || ub < lb + 1e-10
                    break
                end

                # Prune -- could improve the upper bound (TODO: maybe also the lower bound?)
                # println(out, "* Starting with $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
                tic = time_ns()
                # cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, exp(lb), prune_attempts)
                cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, lb, prune_attempts)
                merge!(counters, prune_counters)
                toc = time_ns()
                # println(out, "* Pruning gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
                # update_and_log(cur_root,root,quer,cache,lb,results,i,true, prune_attempts=actual_reps, time=(toc-tic)/1.0e9)

                # Split root (move a quer variable up) -- could improve both the upper and lower bounds
                tic = time_ns()
                to_split = get_to_split(cur_root, splittable, counters, heur, lb)
                cur_root = add_and_split(cur_root, to_split)
                toc = time_ns()
                # println(out, "* Splitting on $(to_split) gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
                delete!(splittable, to_split)
                # ub, lb = update_and_log(cur_root,root,quer,cache,lb,results,i,false, split_var=to_split, time=(toc-tic)/1.0e9)

                log_per_iter(results)
                # TODO: also save the circuit at the end of each iteration for easy retrieval?    
            end
        end
    end
    @assert num_edges(cur_root) == 191438
    @assert num_nodes(cur_root) == 112073
end

Profile.init(n = 10^7, delay = 0.002)
Profile.clear()
@time my_mmap_solve(pc, myquer, num_iter=2, heur="UB", log_per_iter=log_func, out=devnull);

@time mmap_solve(pc, myquer, num_iter=2, heur="UB", log_per_iter=log_func, out=devnull);
@btime mmap_solve($pc, $myquer, num_iter=2, heur="UB", log_per_iter=$log_func, out=devnull);

@profile mmap_solve(pc, myquer, num_iter=2, heur="UB", log_per_iter=log_func, out=devnull);

Profile.print(format=:flat, mincount=100, sortedby=:count)

using ProfileSVG
ProfileSVG.save("prof.svg", timeunit=:ms, yflip=true)
