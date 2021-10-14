using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsFuns: logaddexp
using TikzPictures
using StatsBase: sample
using Random
using ArgParse
using Dates
using JSON
using Serialization
using CSV

function add_basic_arg(s::ArgParseSettings)
    @add_arg_table! s begin
        "prob-circ"
            help = "Path to PSDD file"
            arg_type = String
            required = true
        "--outdir", "-d"
            help = "Output directory"
            arg_type = String
            default = "exp-results/"
        "--seed"
            help = "Seed for random generation"
            arg_type = Int64
            default = rand(1:9999)
        "--quer-percent"
            help = "Percentage of the variables to be randomly selected as query variables (in decimals)"
            arg_type = Float64
            default = 0.5
        "--evid-percent"
            help = "Percentage of the variables to be randomly selected as evidence variables (in decimals)"
            arg_type = Float64
            default = 0.0
        "--qe"
            help = "Path to the file containing query-evidence instances. If specified, ignores --quer-percent and --evid-percent"
            arg_type = String
            default = ""
        "--instance-id"
            help = "Instance ID to read from the query-evidence instances."
            arg_type = Int64
            default = 1
        "--timeout", "-t"
            help = "Timeout in seconds"
            arg_type = Float64
            default = 3600.0
        "--iters"
            help = "Number of maximum iterations of pruning and splitting (defaults to the number of query variables)"
            arg_type = Int64
            default = -1
        "--prune-attempts"
            help = "Number of pruning attempts for each iteration"
            arg_type = Int64
            default = 10
        "--pick-var"
            arg_type = String
            help = "Heuristic method to do split (maxP, minD, depth, rand)"
            default = "maxP"
        "--exp-id"
            help = "Experiment id"
            arg_type = String
            default = Dates.format(now(), "yyyymmdd-HHMMSSs")
        "--only-generate"
            help = "Generate query-evidence instances"
            action = :store_true
        "--out-spn"
            help = "Do not run solver and output the circuit as .spn after marginalizing out non-query variables"
            action = :store_true
        "--verbose", "-v"
            help = "Verbose option. Writes progress.csv"
            action = :store_true
        end
end

function parse_cmd_line()
    s = ArgParseSettings()
    add_basic_arg(s)
    return parse_args(s)
end

function main()
    args = parse_cmd_line()
    circ_path = args["prob-circ"]
    outdir = args["outdir"]
    seed = args["seed"]
    quer_percent = args["quer-percent"]
    evid_percent = args["evid-percent"]
    qe_path = args["qe"]
    id = args["instance-id"]
    timeout = args["timeout"]
    iter = args["iters"]
    prune_attempts = args["prune-attempts"]
    pick_var = args["pick-var"]
    exp_id = args["exp-id"]
    out_spn = args["out-spn"]
    only_generate = args["only-generate"]
    verbose = args["verbose"]

    if only_generate
        if !isdir(joinpath(outdir, "spns"))
            mkpath(joinpath(outdir, "spns"))
        end
        generate_instances(outdir, circ_path, quer_percent, evid_percent, 10)
        return
    end

    out_path = joinpath(outdir, exp_id)
    if !isdir(out_path)
        mkpath(out_path)
    end

    # log config
    open(joinpath(out_path, "config.json"),"w") do f
        write(f, JSON.json(args))
    end

    Random.seed!(seed)
    pc = read(circ_path, ProbCircuit)
    nvars = num_variables(pc)
    vars = variables(pc)

    if isempty(qe_path)
        quer_size = round(Int, quer_percent*nvars)
        quer = BitSet(sample(1:nvars, quer_size, replace=false))
        open(f -> serialize(f,quer), joinpath(out_path, "quer.jls"), "w")

        evid_size = round(Int, evid_percent*nvars)
        if evid_size > 0
            evid_vars = sample(collect(setdiff(variables(pc), quer)), evid_size, replace=false)
            evid_vars_set = BitSet(evid_vars)
            while true
                assignments = bitrand(evid_size)
                evid_dict = Dict(evid_vars .=> assignments)
                df = DataFrame(transpose([x ∈ evid_vars_set ? evid_dict[x] : missing for x in vars]), :auto)
                evid = [assignments[i] ? evid_vars[i] : -evid_vars[i] for i in 1:evid_size]
                if MAR(pc, df)[1] > -Inf
                    break
                end
            end
        else
            evid = []
        end
    else
        qe = CSV.read(qe_path, DataFrame, missingstring="?", skipto=id+1, limit=1)
        quer = BitSet(filter(i -> !ismissing(qe[1,i]) && qe[1,i] == "*", 1:nvars))
        quer_size = length(quer)
        evid = map(i -> qe[1,i] ? i : -i, filter(i -> !ismissing(qe[1,i]) && qe[1,i] isa Bool, 1:nvars))
    end

    if !isempty(evid)
        pc = pc_condition(pc, Int32.(evid)...)
    end

    if out_spn
        spn = marginalize_out(pc, setdiff(variables(pc), quer))
        spn, mapping = make_vars_contiguous(spn)
        write(joinpath(out_path, "$(basename(circ_path)).spn"), spn)
        open(f -> serialize(f,mapping), joinpath(out_path, "mapping.jls"), "w")
        # return
    end

    csv = joinpath(out_path, "progress.csv")
    log_func(results) = begin
        table = DataFrame(;[Symbol(x) => results[x] for x in MMAP_LOG_HEADER]...)
        CSV.write(csv, table; )
    end

    # Do a dummy round of solving to compile everything once
    mini_pc = read("/space/yjchoi/Circuit-Model-Zoo/psdds/little_4var.psdd", ProbCircuit)
    mini_quer = BitSet([1,2,3])
    mmap_solve(mini_pc, mini_quer, num_iter=1)

    @show out_path, seed, iter, prune_attempts, length(quer)
    did_timeout, total_time, iter, ub, lb, lb_state, pc = mmap_solve(pc, quer, 
                    num_iter=(iter < 0 ? quer_size : iter),
                    timeout=timeout,
                    prune_attempts=prune_attempts,
                    log_per_iter=log_func,
                    verbose=verbose,
                    heur=pick_var)

    # Save result
    table = DataFrame("timeout"=>did_timeout, "total_time"=>total_time, "num_iters"=>iter, "ub"=>ub, "lb"=>lb)
    CSV.write(joinpath(out_path, "result.csv"), table)
    CSV.write(joinpath(out_path, "mmap.csv"), DataFrame(transpose(lb_state),:auto), missingstring="?")
    open(f -> serialize(f,pc), joinpath(out_path, "circuit.jls"), "w")

    # read_pc = open(deserialize, joinpath(out_path, "circuit.jls"))
end

function generate_instances(out_path, circ_path, quer_percent, evid_percent, num_instances=100)
    pc = read(circ_path, ProbCircuit)
    nvars = num_variables(pc)
    vars = variables(pc)
    max_var = maximum(vars)
    @assert nvars == max_var
    
    dict = Dict()
    for x in 1:nvars
        dict[x] = []
    end

    for i in 1:num_instances
        quer_size = round(Int, quer_percent*nvars)
        quer = BitSet(sample(1:nvars, quer_size, replace=false))
    
        evid_size = round(Int, evid_percent*nvars)
        evid_vars = sample(collect(setdiff(vars, quer)), evid_size, replace=false)
        evid_vars_set = BitSet(evid_vars)
        assignments = bitrand(evid_size)
        evid = Dict(evid_vars .=> assignments)

        while true
            df = DataFrame(transpose([x ∈ evid_vars_set ? evid[x] : missing for x in vars]), :auto)
            if MAR(pc, df)[1] > -Inf
                break
            end
            assignments = bitrand(evid_size)
            evid = Dict(evid_vars .=> assignments)
        end

        for x in 1:nvars
            if x ∈ quer
                push!(dict[x], "*")
            elseif x ∈ evid_vars_set
                push!(dict[x], evid[x] ? "true" : "false")
            else
                push!(dict[x], "?")
            end
        end

        evid_lits = [assignments[i] ? evid_vars[i] : -evid_vars[i] for i in 1:evid_size]
        pc_cond = pc_condition(pc, Int32.(evid_lits)...)
        spn = marginalize_out(pc_cond, setdiff(vars, quer))
        spn, mapping = make_vars_contiguous(spn)
        write(joinpath(out_path, "spns/$(basename(circ_path))-$(i).spn"), spn)
        open(f -> serialize(f,mapping), joinpath(out_path, "spns/$(basename(circ_path))-$(i)-mapping.jls"), "w")
    end

    csv = joinpath(out_path, "$(basename(circ_path)).csv")
    table = DataFrame(;[Symbol(x) => dict[x] for x in 1:nvars]...)
    CSV.write(csv, table)
end

main()

function some_tests()
    Random.seed!(123)

    prob_circ = load_prob_circuit("/home/tal/Documents/Circuit-Model-Zoo/psdds/asia.uai.psdd")
    quer = BitSet([2,5])
    # Tinker with conditioning
    pc = pc_condition(prob_circ, Var(3), Var(4), Var(7))
    get_margs(pc, 8, [2,5], [1,8])
    pc_cond = pc_condition(pc, Var(1), Var(8))
    get_margs(pc_cond, 8, [2,5], [])

    # Tinker with splitting
    tp = plot(prob_circ)
    save(SVG("pc"), tp)
    splt = add_and_split(prob_circ, Var(2))
    tp = plot(splt)
    save(SVG("split_pc"), tp)
    get_margs(prob_circ, 8, [4,5], [])
    get_margs(splt, 8, [4,5], [])
    get_margs(splt.children[1], 8, [4,5], [])
    get_margs(splt.children[2], 8, [4,5], [])

    # What do we get from splitting?
    prob_circ = load_prob_circuit("/home/tal/Documents/Circuit-Model-Zoo/50-12-10.uai.psdd")
    quer = BitSet(sample(1:num_variables(prob_circ), 72, replace=false))
    quer = BitSet(sample(1:num_variables(prob_circ), 48, replace=false))
    root = mmap_solve(prob_circ, quer, num_iter=5, heur="maxDepth")
    # map_mpe_random(prob_circ, quer)

    # Figure out why renormalizing affects forward bounds
    # to_split = collect(quer)[10]

    # forward_bounds(prob_circ, quer)[prob_circ]
    # num_edges(prob_circ)
    # root = rep_mpe_pruning(prob_circ, quer, 1)
    # forward_bounds(root, quer)[root]
    # num_edges(root)
    # new_and = PlainMulNode([root])
    # new_root = PlainSumNode([new_and])

    # split_root = split(new_root, (new_root, new_and), Var(to_split), callback=keep_params, keep_unary=true)[1]
    # reduce(min, collect(Iterators.flatten(map(x -> x.log_probs, sum_nodes(split_root)))))
    # fb_sp = forward_bounds(split_root, quer)
    # fb_sp[split_root]
    # num_edges(split_root)

    # Circuit should be left unnormalized after pruning

    # norm_split_root = bottomup_renorm_params(split_root)
    # fb_norm = forward_bounds(norm_split_root, quer)
    # fb_norm[norm_split_root]
    # num_edges(norm_split_root)

    # norm_norm_split_root = bottomup_renorm_params(norm_split_root)
    # fb_norm_norm = forward_bounds(norm_norm_split_root, quer)
    # fb_norm_norm[norm_norm_split_root]

    # mapreduce(x -> filter(==(-Inf), x.log_probs), vcat, sum_nodes(norm_split_root))
    # reduce(min, collect(Iterators.flatten(map(x -> x.log_probs, sum_nodes(norm_split_root)))))

    # fb_sp[split_root]
    # fb_norm[norm_split_root]

    # fb_norm[split_root]
    # map(x -> fb_norm[x], split_root.children)
    # map(x -> fb_sp[x], split_root.children)
    # norm_split_root.log_probs
    # exp(norm_split_root.log_probs[1]) + exp(norm_split_root.log_probs[2])
end