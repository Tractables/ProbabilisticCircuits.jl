using ..Utils

"Map from literal to LogicalΔNode"
const LitCache = Dict{Lit, LogicalΔNode}

"Use literal to represent constraint (1 to X, -1 to not X), 0 to represent true"
const ⊤ = convert(Lit, 0)

"""
Learning from data a structured-decomposable circuit with several structure learning algorithms
"""
function learn_struct_prob_circuit(data::Union{XData, WXData}; 
        pseudocount = 1.0, algo = "chow-liu", algo_kwargs=(α=1.0, clt_root="graph_center"), vtree = "chow-liu", vtree_kwargs=(vtree_mode="balanced",))
    if algo == "chow-liu"
        clt = learn_chow_liu_tree(data; algo_kwargs...)
        vtree = learn_vtree_from_clt(clt; vtree_kwargs...);
        pc = compile_psdd_from_clt(clt, vtree);
        estimate_parameters(pc, convert(XBatches,data); pseudocount = pseudocount)
        pc, vtree
    else
        error("Cannot learn a structured-decomposable circuit with algorithm $algo")
    end
end

#############
# Learn PlainVtree from CLT
#############

"
Learn a vtree from clt,
with strategy (close to) `linear` or `balanced`
"
function learn_vtree_from_clt(clt::CLT; vtree_mode::String)::PlainVtree
    roots = [i for (i, x) in enumerate(parent_vector(clt)) if x == 0]
    rootnode = construct_children(Var.(roots), clt, vtree_mode)

    return node2dag(rootnode)
end

function construct_node(v::Var, clt::CLT, strategy::String)::PlainVtreeNode
    children = Var.(outneighbors(clt, v))
    if isempty(children) # leaf node
        return PlainVtreeLeafNode(v)
    else
        right = construct_children(children, clt, strategy)
        return add_parent(v, right)
    end
end

function construct_children(children::Vector{Var}, clt::CLT, strategy::String)::PlainVtreeNode
    sorted_vars = sort(collect(children))
    children_nodes = Vector{PlainVtreeNode}()
    foreach(x -> push!(children_nodes, construct_node(x, clt, strategy)), sorted_vars)

    if strategy == "linear"
        construct_children_linear(children_nodes, clt)
    elseif strategy == "balanced"
        construct_children_balanced(children_nodes, clt)
    else
        throw("Unknown type of strategy")
    end
end

function construct_children_linear(children_nodes::Vector{PlainVtreeNode}, clt::CLT)::PlainVtreeNode
    children_nodes = Iterators.Stateful(reverse(children_nodes))

    right = popfirst!(children_nodes)
    for left in children_nodes
        right = PlainVtreeInnerNode(left, right)
    end
    return right
end

function construct_children_balanced(children_nodes::Vector{PlainVtreeNode}, clt::CLT)::PlainVtreeNode
    if length(children_nodes) == 1
        return children_nodes[1]
    elseif length(children_nodes) == 2
        return PlainVtreeInnerNode(children_nodes[1], children_nodes[2])
    else
        len = trunc(Int64, length(children_nodes) / 2)
        left = construct_children_balanced(children_nodes[1 : len], clt)
        right = construct_children_balanced(children_nodes[len + 1 : end], clt)
        return PlainVtreeInnerNode(left, right)
    end
end

function add_parent(parent::Var, children::PlainVtreeNode)
    return PlainVtreeInnerNode(PlainVtreeLeafNode(parent), children)
end

#####################
# Compile PSDD from CLT and vtree
#####################

"Compile a psdd circuit from clt and vtree"
function compile_psdd_from_clt(clt::MetaDiGraph, vtree::PlainVtree)
    order = node2dag(vtree[end])
    parent_clt = Var.(parent_vector(clt))

    lin = Vector{ProbΔNode}()
    prob_cache = ProbCache()
    lit_cache = LitCache()
    v2p = Dict{PlainVtreeNode, ProbΔ}()

    get_params(cpt::Dict) = length(cpt) == 2 ? [cpt[1], cpt[0]] : [cpt[(1,1)], cpt[(0,1)], cpt[(1,0)], cpt[(0,0)]]
    function add_mapping!(v::PlainVtreeNode, circuits::ProbΔ)
        if !haskey(v2p, v); v2p[v] = Vector{ProbΔNode}(); end
        foreach(c -> if !(c in v2p[v]) push!(v2p[v], c);end, circuits)
    end

    # compile vtree leaf node to terminal/true node
    function compile_from_vtree_node(v::PlainVtreeLeafNode)
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
    function compile_from_vtree_node(v::PlainVtreeInnerNode)
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
            throw("PlainVtree are not learned from the same CLT")
        end
        add_mapping!(v, circuit)
    end

    foreach(compile_from_vtree_node, vtree)
    return lin
end

#####################
# Construct probabilistic circuit node
#####################

prob_children(n, prob_cache) =  
    copy_with_eltype(map(c -> prob_cache[c], n.children), ProbΔNode{<:StructLogicalΔNode})

"Add leaf nodes to circuit `lin`"
function add_prob_leaf_node(var::Var, vtree::PlainVtreeLeafNode, lit_cache::LitCache, prob_cache::ProbCache, lin)
    pos = StructLiteralNode{PlainVtreeNode}( var2lit(var), vtree)
    neg = StructLiteralNode{PlainVtreeNode}(-var2lit(var), vtree)
    lit_cache[var2lit(var)] = pos
    lit_cache[-var2lit(var)] = neg
    pos2 = ProbLiteral(pos)
    neg2 = ProbLiteral(neg)
    prob_cache[pos] = pos2
    prob_cache[neg] = neg2
    push!(lin, pos2)
    push!(lin, neg2)
    return (pos2, neg2)
end

"Add prob⋀ node to circuit `lin`"
function add_prob⋀_node(children::ProbΔ, vtree::PlainVtreeInnerNode, prob_cache::ProbCache, lin)::Prob⋀
    logic = Struct⋀Node{PlainVtreeNode}([c.origin for c in children], vtree)
    prob = Prob⋀(logic, prob_children(logic, prob_cache))
    prob_cache[logic] = prob
    push!(lin, prob)
    return prob
end

"Add prob⋁ node to circuit `lin`"
function add_prob⋁_node(children::ProbΔ, vtree::PlainVtreeNode, thetas::Vector{Float64}, prob_cache::ProbCache, lin)::Prob⋁
    logic = Struct⋁Node{PlainVtreeNode}([c.origin for c in children], vtree)
    prob = Prob⋁(logic, prob_children(logic, prob_cache))
    prob.log_thetas = log.(thetas)
    prob_cache[logic] = prob
    push!(lin, prob)
    return prob
end

"Construct decision nodes given `primes` and `subs`"
function compile_decision_node(primes::ProbΔ, subs::ProbΔ, vtree::PlainVtreeInnerNode, params::Vector{Float64}, prob_cache::ProbCache, lin)
    elements = [add_prob⋀_node([prime, sub], vtree, prob_cache, lin) for (prime, sub) in zip(primes, subs)]
    return add_prob⋁_node(elements, vtree, params, prob_cache, lin)
end

"Construct literal nodes given variable `var`"
function compile_literal_nodes(var::Var, vtree::PlainVtreeLeafNode, probs::Vector{Float64}, lit_cache::LitCache, prob_cache::ProbCache, lin)
    (pos, neg) = add_prob_leaf_node(var, vtree, lit_cache, prob_cache, lin)
    return [pos, neg]
end

"Construct true nodes given variable `var`"
function compile_true_nodes(var::Var, vtree::PlainVtreeLeafNode, probs::Vector{Float64}, lit_cache::LitCache, prob_cache::ProbCache, lin)
    (pos, neg) = add_prob_leaf_node(var, vtree, lit_cache, prob_cache, lin)
    return [add_prob⋁_node([pos, neg], vtree, probs[i:i+1], prob_cache, lin) for i in 1:2:length(probs)]
end

"Construct decision nodes conditiond on different distribution"
function compile_decision_nodes(primes::ProbΔ, subs::ProbΔ, vtree::PlainVtreeInnerNode, params::Vector{Float64}, prob_cache::ProbCache, lin)
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

function calculate_all_bases(circuit::ProbΔ)::BaseCache
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

function compile_fully_factorized_psdd_from_vtree(vtree::PlainVtree)::ProbΔ

    function ful_factor_node(v::PlainVtreeLeafNode, lit_cache::LitCache, prob_cache::ProbCache, v2n, lin)
        var = variables(v)[1]
        pos, neg = add_prob_leaf_node(var, v, lit_cache, prob_cache, lin)
        prob_or = add_prob⋁_node([pos, neg], v, [0.5, 0.5], prob_cache, lin)
        v2n[v] = prob_or
        nothing
    end

    function ful_factor_node(v::PlainVtreeInnerNode, lit_cache::LitCache, prob_cache::ProbCache, v2n, lin)
        left = v2n[v.left]
        right = v2n[v.right]
        prob_and = add_prob⋀_node([left, right], v, prob_cache, lin)
        prob_or = add_prob⋁_node([prob_and], v, [1.0], prob_cache, lin)
        v2n[v] = prob_or
        nothing
    end

    lin = Vector{ProbΔNode}()
    prob_cache = ProbCache()
    lit_cache = LitCache()
    v2n = Dict{PlainVtreeNode, ProbΔNode}()

    for v in vtree
        ful_factor_node(v, lit_cache, prob_cache, v2n, lin)
    end

    lin
end
