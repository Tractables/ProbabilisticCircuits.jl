using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsFuns: logaddexp
using TikzPictures
using StatsBase: sample

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