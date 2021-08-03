using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsFuns: logaddexp
using TikzPictures
using StatsBase: sample
using Random

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
map_mpe_random(prob_circ, quer)

# Figure out why renormalizing affects forward bounds
to_split = collect(quer)[10]

forward_bounds(prob_circ, quer)[prob_circ]
num_edges(prob_circ)
root = rep_mpe_pruning(prob_circ, quer, 1)
forward_bounds(root, quer)[root]
num_edges(root)
new_and = PlainMulNode([root])
new_root = PlainSumNode([new_and])

split_root = split(new_root, (new_root, new_and), Var(to_split), callback=keep_params, keep_unary=true)[1]
reduce(min, collect(Iterators.flatten(map(x -> x.log_probs, sum_nodes(split_root)))))
fb_sp = forward_bounds(split_root, quer)
fb_sp[split_root]
num_edges(split_root)

# norm_split_root = 
norm_split_root = bottomup_renorm_params(split_root)
fb_norm = forward_bounds(norm_split_root, quer)
fb_norm[norm_split_root]
num_edges(norm_split_root)

norm_norm_split_root = bottomup_renorm_params(norm_split_root)
fb_norm_norm = forward_bounds(norm_norm_split_root, quer)
fb_norm_norm[norm_norm_split_root]

mapreduce(x -> filter(==(-Inf), x.log_probs), vcat, sum_nodes(norm_split_root))
reduce(min, collect(Iterators.flatten(map(x -> x.log_probs, sum_nodes(norm_split_root)))))

fb_sp[split_root]
fb_norm[norm_split_root]

fb_norm[split_root]
map(x -> fb_norm[x], split_root.children)
map(x -> fb_sp[x], split_root.children)
norm_split_root.log_probs
exp(norm_split_root.log_probs[1]) + exp(norm_split_root.log_probs[2])