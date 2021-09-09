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
        "--iters"
            help = "Number of maximum iterations of pruning and splitting"
            arg_type = Int64
            default = 50
        "--prune-attempts"
            help = "Number of pruning attempts for each iteration"
            arg_type = Int64
            default = 10
        "--pick-var"
            arg_type = String
            help = "Heuristic method to do split (maxP, minD, depth)"
            default = "maxP"
        "--exp-id"
            help = "Experiment id"
            arg_type = String
            default = Dates.format(now(), "yyyymmdd-HHMMSSs")
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
    iter = args["iters"]
    prune_attempts = args["prune-attempts"]
    pick_var = args["pick-var"]
    exp_id = args["exp-id"]
    
    # log config
    out_path = joinpath(outdir, exp_id)
    if !isdir(out_path)
        mkpath(out_path)
    end
    open(joinpath(out_path, "config.json"),"w") do f
        write(f, JSON.json(args))
    end

    Random.seed!(seed)
    pc = load_prob_circuit(circ_path)
    quer = BitSet(sample(1:num_variables(pc), round(Int,quer_percent*num_variables(pc)), replace=false))

    csv = joinpath(out_path, "progress.csv")
    log_func(results) = begin
        table = DataFrame(;[Symbol(x) => results[x] for x in MMAP_LOG_HEADER]...)
        CSV.write(csv, table; )
    end

    @show out_path, seed, iter, prune_attempts, length(quer)
    pc = mmap_solve(pc, quer, 
                    num_iter=iter,
                    prune_attempts=prune_attempts,
                    log_per_iter=log_func,
                    heur=pick_var)
    
    # Save result
    open(f -> serialize(f,pc), joinpath(out_path, "circuit.jls"), "w")
    # read_pc = open(deserialize, joinpath(out_path, "circuit.jls"))
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