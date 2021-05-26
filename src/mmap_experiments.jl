using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsFuns: logaddexp
using TikzPictures

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
get_margs(pc, 8, [4,5], [])
get_margs(splt, 8, [4,5], [])

cond_circ = pc_condition(prob_circ, Var(1))
ccond_circ = pc_condition(cond_circ, Var(8))
norm_cond_circ = bottomup_renorm_params(ccond_circ)
@show cond_distr = get_margs(norm_cond_circ, 8, [2,5], [])
@show mar_distr = get_margs(prob_circ, 8, [2,5], [1,8])
distr2 = get_margs(cond_circ, 8, [2,5],[])

num_edges(prob_circ)
num_edges(cond_circ)
prob_circ.children[1].children[2].log_probs
cond_circ.children[1].children[2].log_probs

norm_pc = normalize_params(prob_circ)
df = DataFrame(missings(Bool, 1, num_variables(prob_circ)))
@show marginal_log_likelihood(prob_circ, df)
@show marginal_log_likelihood(norm_pc, df)
@show marginal_log_likelihood(ccond_circ.children[2].children[2], df)

map(x -> num_children(x), innernodes(ccond_circ))
ru_ccond_circ = remove_unary_gates(ccond_circ)
@show distr4 = get_margs(ccond_circ, 8, [1,8], [])

@show prob_circ.children[2].children[2].log_probs
@show ccond_circ.children[2].children[2].log_probs
variables(prob_circ.children[2].children[2])
variables(ccond_circ.children[2].children[2])
num_edges(prob_circ.children[2].children[2])
num_edges(ccond_circ.children[2].children[2])
@show get_margs(prob_circ.children[2].children[2].children[1], 8, [8], [3,5])
@show get_margs(ccond_circ.children[2].children[2].children[1], 8, [8], [])
@show get_margs(prob_circ.children[2].children[2].children[2], 8, [8], [3,5])
@show get_margs(ccond_circ.children[2].children[2].children[2], 8, [8], [])
@show get_margs(prob_circ.children[2].children[2], 8, [8], [3,5])
@show get_margs(ccond_circ.children[2].children[2], 8, [8], [])
norm_ccond_circ = bottomup_renorm_params(ccond_circ)
@show get_margs(norm_ccond_circ.children[2].children[2], 8, [8], [])

map(x -> reduce(logaddexp, x.log_probs), sum_nodes(norm_ccond_circ))

ccond_circ.children[2].children[2].children[1].children
ccond_circ.log_probs
prob_circ.children[1].children[2].log_probs
ccond_circ.children[1].children[2].log_probs

cond_distr2 = get_margs(prob_circ, [1,8], [5])
@show cond_distr2 .- reduce(logaddexp, cond_distr2)
@show distr3 = get_margs(cond_circ2, [1,8], [2])


split_circ = add_and_split(prob_circ, Var(4))
split_circ.log_probs
split_circ.children
split_circ.children[1].log_probs
split_circ.children[2].log_probs
prob_circ.log_probs
split_circ.children[2].children[2].children[2].log_probs
prob_circ.children[2].children[2].log_probs

num_children(split_circ.children[1])
split_circ.log_probs
brute_force_mmap(split_circ, quer)
brute_force_mmap(split_circ.children[1], quer)
brute_force_mmap(split_circ.children[2], quer)
brute_force_mmap(prob_circ.children[1], quer)
brute_force_mmap(prob_circ.children[2], quer)
all(map(x -> length(x.children) == length(x.log_probs), sum_nodes(split_circ)))
brute_force_mmap(prob_circ, quer)
respects_vtree(split_circ,infer_vtree(prob_circ))
data = Dict()
impl_lits = implied_literals(prob_circ, data)
data[prob_circ.children[2]]
impl_lits[split_circ]
impl_lits[split_circ.children[1]]
impl_lits[split_circ.children[2]]
associated_with_mult(split_circ, quer, impl_lits)
num_edges(prob_circ)
num_edges(split_circ.children[1])
num_edges(split_circ)

forward_bounds(split_circ, quer)[split_circ]