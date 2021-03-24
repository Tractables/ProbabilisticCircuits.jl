using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits

prob_circ = load_prob_circuit("/home/tal/Documents/Circuit-Model-Zoo/fgs/asia.uai.psdd")
quer = BitSet([1,5])
split_circ = add_and_split(prob_circ, Var(5))
test = remove_unary_gates(split_circ)
test.children
test.children[1]
test.children[2]
test.log_probs
split_circ.children
split_circ.children[1]
num_children(split_circ.children[1])
test.log_probs
split_circ.log_probs
brute_force_mmap(test, quer)
brute_force_mmap(prob_circ, quer)
split_circ2 = add_and_split(test, Var(1))
test = remove_unary_gates(test)
forward_bounds(test, quer)[test]