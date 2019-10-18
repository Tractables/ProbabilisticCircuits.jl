using Printf
using HDF5

"Map from literal to LogicalCircuitNode"
const LitCache = Dict{Lit, LogicalCircuitNode}

"Use literal to represent constraint (1 to X, -1 to not X), 0 to represent true"
const ⊤ = convert(Lit, 0)

"Map logical variable to bases"
const BaseCache = Dict{LogicalCircuitNode, Vector{Lit}}

"Cache circuit node parents "
const ParentsCache = Dict{ProbCircuitNode,Vector{<:ProbCircuitNode}}

"Wrapper for PSDD, cache structures used in learning process"
mutable struct PSDDWrapper
    pc::ProbCircuit△
    bases::BaseCache
    parents::ParentsCache
    vtree::Vtree△
    PSDDWrapper(pc, bases, parents, vtree) = new(pc, bases, parents, vtree)
end

import Base.length
@inline length(psdd::PSDDWrapper) = length(psdd.pc) 

"""
Builds a Chow-Liu tree from complete data
"""
function build_clt_structure(data::PlainXData; 
                            vtree_mode="balanced",
                            clt_kwargs=(α=1.0,
                                        parametered=true,
                                        clt_root="graph_center"))::PSDDWrapper
    clt = learn_chow_liu_tree(data; clt_kwargs...);
    vtree = learn_vtree_from_clt(clt; vtree_mode=vtree_mode);
    pc, bases = compile_psdd_from_clt(clt, vtree);
    parents = parents_vector(pc)
    return PSDDWrapper(pc, bases, parents, vtree)
end

function build_rand_structure(data::PlainXData; vtree_mode="rand")::PSDDWrapper
    vtree = random_vtree(num_features(data); vtree_mode="rand") # TODO: add interface later
    pc = compile_fully_factorized_psdd_from_vtree(vtree)
    bases = calculate_all_bases(pc)
    parents = parents_vector(pc)
    return PSDDWrapper(pc, bases, parents, vtree)
end

function build_bottom_up_structure(data::PlainXData; α)::PSDDWrapper
    vtree = learn_vtree_bottom_up(data;α=α)
    pc = compile_fully_factorized_psdd_from_vtree(vtree)
    bases = calculate_all_bases(pc)
    parents = parents_vector(pc)
    return PSDDWrapper(pc, bases, parents, vtree)
end

#############
# Learn Vtree from CLT
#############

"
Learn a vtree from clt,
with strategy (close to) `linear` or `balanced`
"
function learn_vtree_from_clt(clt::CLT; vtree_mode::String)::Vtree△
    roots = [i for (i, x) in enumerate(parent_vector(clt)) if x == 0]
    root = construct_children(Var.(roots), clt, vtree_mode)

    return order_nodes_leaves_before_parents(root)
end

function construct_node(v::Var, clt::CLT, strategy::String)::VtreeNode
    children = Var.(outneighbors(clt, v))
    if isempty(children) # leaf node
        return VtreeLeafNode(v)
    else
        right = construct_children(children, clt, strategy)
        return add_parent(v, right)
    end
end

function construct_children(children::Vector{Var}, clt::CLT, strategy::String)::VtreeNode
    sorted_vars = sort(collect(children))
    children_nodes = Vector{VtreeNode}()
    foreach(x -> push!(children_nodes, construct_node(x, clt, strategy)), sorted_vars)

    if strategy == "linear"
        construct_children_linear(children_nodes, clt)
    elseif strategy == "balanced"
        construct_children_balanced(children_nodes, clt)
    else
        throw("Unknown type of strategy")
    end
end

function construct_children_linear(children_nodes::Vector{VtreeNode}, clt::CLT)::VtreeNode
    children_nodes = Iterators.Stateful(reverse(children_nodes))

    right = popfirst!(children_nodes)
    for left in children_nodes
        right = VtreeInnerNode(left, right)
    end
    return right
end

function construct_children_balanced(children_nodes::Vector{VtreeNode}, clt::CLT)::VtreeNode
    if length(children_nodes) == 1
        return children_nodes[1]
    elseif length(children_nodes) == 2
        return VtreeInnerNode(children_nodes[1], children_nodes[2])
    else
        len = trunc(Int64, length(children_nodes) / 2)
        left = construct_children_balanced(children_nodes[1 : len], clt)
        right = construct_children_balanced(children_nodes[len + 1 : end], clt)
        return VtreeInnerNode(left, right)
    end
end

function add_parent(parent::Var, children::VtreeNode)
    return VtreeInnerNode(VtreeLeafNode(parent), children)
end

#####################
# Compile PSDD from CLT and vtree
#####################

"Compile a psdd circuit from clt and vtree"
function compile_psdd_from_clt(clt::MetaDiGraph, vtree::Vtree△)
    order = order_nodes_leaves_before_parents(vtree[end])
    parent_clt = Var.(parent_vector(clt))

    lin = Vector{ProbCircuitNode}()
    prob_cache = ProbCache()
    lit_cache = LitCache()
    v2p = Dict{VtreeNode, ProbCircuit△}()

    get_params(cpt::Dict) = length(cpt) == 2 ? [cpt[1], cpt[0]] : [cpt[(1,1)], cpt[(0,1)], cpt[(1,0)], cpt[(0,0)]]
    function add_mapping!(v::VtreeNode, circuits::ProbCircuit△)
        if !haskey(v2p, v); v2p[v] = Vector{ProbCircuitNode}(); end
        foreach(c -> if !(c in v2p[v]) push!(v2p[v], c);end, circuits)
    end

    # compile vtree leaf node to terminal/true node
    function compile_from_vtree_node(v::VtreeLeafNode)
        var = v.var
        children = Var.(outneighbors(clt, var))
        cpt = get_prop(clt, var, :cpt)
        parent = parent_clt[var]
        if isequal(children, [])
            circuit = compile_true_nodes(var, v, get_params(cpt), lit_cache, prob_cache, lin)
        else
            circuit = compile_literal_nodes(var, v, get_params(cpt), lit_cache, prob_cache, lin)
        end
        add_mapping!(v, circuit)
    end

    # compile to decision node
    function compile_from_vtree_node(v::VtreeInnerNode)
        left_var = left_most_child(v.left).var
        right_var = left_most_child(v.right).var
        left_circuit = v2p[v.left]
        right_circuit = v2p[v.right]

        if parent_clt[left_var] == parent_clt[right_var] # two nodes are independent, compile to seperate decision nodes
            circuit = [compile_decision_node([l], [r], v, [1.0], prob_cache, lin) for (l, r) in zip(left_circuit, right_circuit)]
        elseif left_var == parent_clt[right_var] # conditioned on left
            cpt = get_prop(clt, left_var, :cpt)
            circuit = compile_decision_nodes(left_circuit, right_circuit, v, get_params(cpt), prob_cache, lin)
        else
            throw("Vtree are not learned from the same CLT")
        end
        add_mapping!(v, circuit)
    end

    foreach(compile_from_vtree_node, vtree)
    bases = calculate_all_bases(lin)
    return (lin, bases)
end

#####################
# Construct probabilistic circuit node
#####################

prob_children(n, prob_cache) =  
    copy_with_eltype(map(c -> prob_cache[c], n.children), ProbCircuitNode{<:StructLogicalCircuitNode})

"Add leaf nodes to circuit `lin`"
function add_prob_leaf_node(var::Var, vtree::VtreeLeafNode, lit_cache::LitCache, prob_cache::ProbCache, lin)::Tuple{ProbLiteral, ProbLiteral}
    pos = StructLiteralNode( var2lit(var), vtree)
    neg = StructLiteralNode(-var2lit(var), vtree)
    lit_cache[var2lit(var)] = pos
    lit_cache[-var2lit(var)] = neg
    pos2 = ProbLiteral{StructLiteralNode}(pos)
    neg2 = ProbLiteral{StructLiteralNode}(neg)
    prob_cache[pos] = pos2
    prob_cache[neg] = neg2
    push!(lin, pos2)
    push!(lin, neg2)
    return (pos2, neg2)
end

"Add prob⋀ node to circuit `lin`"
function add_prob⋀_node(children::ProbCircuit△, vtree::VtreeInnerNode, prob_cache::ProbCache, lin)::Prob⋀
    logic = Struct⋀Node([c.origin for c in children], vtree)
    prob = Prob⋀{StructLogicalCircuitNode}(logic, prob_children(logic, prob_cache))
    prob_cache[logic] = prob
    push!(lin, prob)
    return prob
end

"Add prob⋁ node to circuit `lin`"
function add_prob⋁_node(children::ProbCircuit△, vtree::VtreeNode, thetas::Vector{Float64}, prob_cache::ProbCache, lin)::Prob⋁
    logic = Struct⋁Node([c.origin for c in children], vtree)
    prob = Prob⋁(StructLogicalCircuitNode, logic, prob_children(logic, prob_cache))
    prob.log_thetas = log.(thetas)
    prob_cache[logic] = prob
    push!(lin, prob)
    return prob
end

"Construct decision nodes given `primes` and `subs`"
function compile_decision_node(primes::ProbCircuit△, subs::ProbCircuit△, vtree::VtreeInnerNode, params::Vector{Float64}, prob_cache::ProbCache, lin)
    elements = [add_prob⋀_node([prime, sub], vtree, prob_cache, lin) for (prime, sub) in zip(primes, subs)]
    return add_prob⋁_node(elements, vtree, params, prob_cache, lin)
end

"Construct literal nodes given variable `var`"
function compile_literal_nodes(var::Var, vtree::VtreeLeafNode, probs::Vector{Float64}, lit_cache::LitCache, prob_cache::ProbCache, lin)
    (pos, neg) = add_prob_leaf_node(var, vtree, lit_cache, prob_cache, lin)
    return [pos, neg]
end

"Construct true nodes given variable `var`"
function compile_true_nodes(var::Var, vtree::VtreeLeafNode, probs::Vector{Float64}, lit_cache::LitCache, prob_cache::ProbCache, lin)
    (pos, neg) = add_prob_leaf_node(var, vtree, lit_cache, prob_cache, lin)
    return [add_prob⋁_node([pos, neg], vtree, probs[i:i+1], prob_cache, lin) for i in 1:2:length(probs)]
end

"Construct decision nodes conditiond on different distribution"
function compile_decision_nodes(primes::ProbCircuit△, subs::ProbCircuit△, vtree::VtreeInnerNode, params::Vector{Float64}, prob_cache::ProbCache, lin)
    return [compile_decision_node(primes, subs, vtree, params[i:i+1], prob_cache, lin) for i in 1:2:length(params)]
end

#####################
# Map and cache constraints
#####################

function set_base(index, n::StructLiteralNode, bases)
    if positive(n)
        bases[n][variable(n)] = 1
    else
        bases[n][variable(n)] = -1
    end
end

function set_base(index, n::Struct⋁Node, bases)
    len = num_children(n)
    temp = sum([bases[c] for c in n.children])
    bases[n] = map(x-> if x == len 1; elseif -x == len; -1; else 0; end, temp)
end

function set_base(index, n::Struct⋀Node, bases)
    bases[n] = sum([bases[c] for c in n.children])
end

function calculate_all_bases(circuit::ProbCircuit△)::BaseCache
    num_var = num_variables(circuit[end].origin.vtree)
    bases = BaseCache()
    foreach(n -> bases[n.origin] = fill(⊤, num_var), circuit)
    foreach(n -> set_base(n[1], n[2].origin, bases), enumerate(circuit))
    @assert all(bases[circuit[end].origin] .== ⊤) "Base of root node should be true"
    return bases
end

#####################
# Compile fully factorized PSDD from vtree, all variables are independent initially
#####################

function compile_fully_factorized_psdd_from_vtree(vtree::Vtree△)::ProbCircuit△

    function ful_factor_node(v::VtreeLeafNode, lit_cache::LitCache, prob_cache::ProbCache, v2n, lin)
        var = variables(v)[1]
        pos, neg = add_prob_leaf_node(var, v, lit_cache, prob_cache, lin)
        prob_or = add_prob⋁_node([pos, neg], v, [0.5, 0.5], prob_cache, lin)
        v2n[v] = prob_or
        nothing
    end

    function ful_factor_node(v::VtreeInnerNode, lit_cache::LitCache, prob_cache::ProbCache, v2n, lin)
        left = v2n[v.left]
        right = v2n[v.right]
        prob_and = add_prob⋀_node([left, right], v, prob_cache, lin)
        prob_or = add_prob⋁_node([prob_and], v, [1.0], prob_cache, lin)
        v2n[v] = prob_or
        nothing
    end

    lin = Vector{ProbCircuitNode}()
    prob_cache = ProbCache()
    lit_cache = LitCache()
    v2n = Dict{VtreeNode, ProbCircuitNode}()

    for v in vtree
        ful_factor_node(v, lit_cache, prob_cache, v2n, lin)
    end

    lin
end