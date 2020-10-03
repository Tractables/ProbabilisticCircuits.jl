export learn_chow_liu_tree_circuit, learn_vtree_from_clt, compile_sdd_from_clt
using LightGraphs: outneighbors
using MetaGraphs: get_prop

"""
Learning from data a structured-decomposable circuit with several structure learning algorithms
"""
function learn_chow_liu_tree_circuit(data;
        pseudocount = 1.0, algo_kwargs=(α=1.0, clt_root="graph_center"), vtree_kwargs=(vtree_mode="balanced",))    
    clt = learn_chow_liu_tree(data; algo_kwargs...)
    vtree = learn_vtree_from_clt(clt; vtree_kwargs...)
    lc = compile_sdd_from_clt(clt, vtree)
    pc = ProbCircuit(lc)
    estimate_parameters(pc, data; pseudocount=pseudocount)
    pc, vtree
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

    return rootnode
end

function construct_node(v::Var, clt::CLT, strategy::String)::PlainVtree
    children = Var.(outneighbors(clt, v))
    if isempty(children) # leaf node
        return PlainVtreeLeafNode(v)
    else
        right = construct_children(children, clt, strategy)
        return add_parent(v, right)
    end
end

function construct_children(children::Vector{Var}, clt::CLT, strategy::String)::PlainVtree
    sorted_vars = sort(collect(children))
    children_nodes = Vector{PlainVtree}()
    foreach(x -> push!(children_nodes, construct_node(x, clt, strategy)), sorted_vars)

    if strategy == "linear"
        construct_children_linear(children_nodes, clt)
    elseif strategy == "balanced"
        construct_children_balanced(children_nodes, clt)
    else
        throw("Unknown type of strategy")
    end
end

function construct_children_linear(children_nodes::Vector{PlainVtree}, clt::CLT)::PlainVtree
    children_nodes = Iterators.Stateful(reverse(children_nodes))

    right = popfirst!(children_nodes)
    for left in children_nodes
        right = PlainVtreeInnerNode(left, right)
    end
    return right
end

function construct_children_balanced(children_nodes::Vector{PlainVtree}, clt::CLT)::PlainVtree
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

function add_parent(parent::Var, children::PlainVtree)
    return PlainVtreeInnerNode(PlainVtreeLeafNode(parent), children)
end

#####################
# Compile PSDD from CLT and vtree
#####################

"Compile a psdd circuit from clt and vtree"
function compile_sdd_from_clt(clt::CLT, vtree::PlainVtree)::PlainStructLogicCircuit

    parent_clt = Var.(parent_vector(clt))
    v2p = Dict{PlainVtree, Vector{PlainStructLogicCircuit}}()
    
    function add_mapping!(v::PlainVtree, circuits)
        if !haskey(v2p, v); v2p[v] = Vector{PlainStructLogicCircuit}(); end
        foreach(c -> if !(c in v2p[v]) push!(v2p[v], c);end, circuits)
    end

    # compile vtree leaf node to terminal/true node
    function compile_from_vtree_node(v::PlainVtreeLeafNode)
        var = v.var
        children = Var.(outneighbors(clt, var))
        cpt = get_prop(clt, var, :cpt)
        parent = parent_clt[var]
        if isequal(children, [])
            circuit = compile_true_nodes(var, v; num=length(cpt) ÷ 2)
        else
            circuit = compile_canonical_literals(var, v)
        end
        add_mapping!(v, circuit)
        nothing
    end

    # compile to decision node
    function compile_from_vtree_node(v::PlainVtreeInnerNode)
        left_var = left_most_descendent(v.left).var
        right_var = left_most_descendent(v.right).var
        left_circuit = v2p[v.left]
        right_circuit = v2p[v.right]

        if parent_clt[left_var] == parent_clt[right_var] # two nodes are independent, compile to seperate decision nodes
            circuit = [compile_decision_node([l], [r], v) for (l, r) in zip(left_circuit, right_circuit)]
        elseif left_var == parent_clt[right_var] # conditioned on left
            cpt = get_prop(clt, left_var, :cpt)
            circuit = compile_decision_nodes(left_circuit, right_circuit, v; num=length(cpt) ÷ 2)
        else
            throw("PlainVtree are not learned from the same CLT")
        end
        add_mapping!(v, circuit)
        nothing
    end

    foreach(compile_from_vtree_node, vtree)

    v2p[vtree][end]
end

#####################
# Construct circuit node
#####################
"Construct decision nodes given `primes` and `subs`"
function compile_decision_node(primes::Vector{<:PlainStructLogicCircuit}, subs::Vector{<:PlainStructLogicCircuit}, vtree::PlainVtreeInnerNode)
    elements = [conjoin(prime, sub; use_vtree=vtree) for (prime, sub) in zip(primes, subs)]
    return disjoin(elements; use_vtree=vtree)
end

"Construct literal nodes given variable `var`"
function compile_canonical_literals(var::Var, vtree::PlainVtreeLeafNode)
    return [PlainStructLiteralNode( var2lit(var), vtree), PlainStructLiteralNode(-var2lit(var), vtree)]
end

"Construct true nodes given variable `var`"
function compile_true_nodes(var::Var, vtree::PlainVtreeLeafNode; num)
    pos, neg = compile_canonical_literals(var, vtree)
    return [disjoin([pos, neg]; use_vtree = vtree) for _ in 1 : num]
end

"Construct decision nodes conditiond on different distribution"
function compile_decision_nodes(primes::Vector{<:PlainStructLogicCircuit}, subs::Vector{<:PlainStructLogicCircuit}, vtree::PlainVtreeInnerNode; num)
    return [compile_decision_node(primes, subs, vtree) for _ in 1 : num]
end
