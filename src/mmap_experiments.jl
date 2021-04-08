using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits

prob_circ = load_prob_circuit("/home/tal/Documents/Circuit-Model-Zoo/psdds/asia.uai.psdd")
quer = BitSet([1,8])
split_circ = add_and_split(prob_circ, Var(3))
split_circ.children
split_circ.children[1]
num_children(split_circ.children[1])
split_circ.log_probs
brute_force_mmap(split_circ, quer)
brute_force_mmap(prob_circ, quer)
respects_vtree(split_circ,infer_vtree(prob_circ))
impl_lits = implied_literals(split_circ)
associated_with_mult(split_circ, quer, impl_lits)
num_edges(prob_circ)
num_edges(split_circ.children[1])
num_edges(split_circ)