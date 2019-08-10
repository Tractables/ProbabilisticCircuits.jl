using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

"convert literal+/- to probability value 0/1"
@inline lit2value(l::Lit)::Int = (l > 0 ? 1 : 0)


"build probability circuits from Baysian tree "
function compile_prob_circuit_from_clt(clt::MetaDiGraph)::ProbCircuit△
    topo_order = Var.(reverse(topological_sort_by_dfs(clt))) #order to parse the node
    lin = Vector{ProbCircuitNode}()
    node_cache = Dict{Lit, LogicalCircuitNode}()
    prob_cache = ProbCache()

    "default order of circuit node, from left to right: +/1 -/0"

    "compile leaf node into circuits"
    function compile_leaf(ln::Var)
        pos = PosLeafNode(ln)
        neg = NegLeafNode(ln)
        node_cache[var2lit(ln)] = pos
        node_cache[-var2lit(ln)] = neg
        pos = ProbCircuitNode(pos, prob_cache)
        neg = ProbCircuitNode(neg, prob_cache)
        push!(lin, pos)
        push!(lin, neg)
    end

    "compile inner disjunction node"
    function compile_⋁inner(ln::Lit, children::Vector{Var})::Vector{⋁Node}
        logical_nodes = Vector{⋁Node}()
        v = lit2value(ln)

        for c in children
            #build logical ciruits
            temp = ⋁Node([node_cache[lit] for lit in [var2lit(c), - var2lit(c)]])
            push!(logical_nodes, temp)

            n = ProbCircuitNode(temp, prob_cache)

            #calculate weights for each child
            cpt = get_prop(clt, c, :cpt)
            weights = [cpt[(1, v)], cpt[(0, v)]]
            n.log_thetas = log.(weights)
            push!(lin, n)
        end

        return logical_nodes
    end

    "compile inner conjunction node into circuits, left node is indicator, rest nodes are disjunction children nodes"
    function compile_⋀inner(indicator::Lit, children::Vector{⋁Node})
        leaf = node_cache[indicator]
        temp = ⋀Node(vcat([leaf], children))
        node_cache[indicator] = temp

        n = ProbCircuitNode(temp, prob_cache)
        push!(lin, n)
    end

    "compile inner node, 1 inner variable to 2 leaf nodes, 2 * num_children disjunction nodes and 2 conjunction nodes"
    function compile_inner(ln::Var, children::Vector{Var})
        compile_leaf(ln)
        pos⋁ = compile_⋁inner(var2lit(ln), children)
        neg⋁ = compile_⋁inner(-var2lit(ln), children)
        compile_⋀inner(var2lit(ln), pos⋁)
        compile_⋀inner(-var2lit(ln), neg⋁)
    end

    "compile root, add another disjunction node"
    function compile_root(root::Var)
        temp = ⋁Node([node_cache[s] for s in [var2lit(root), -var2lit(root)]])
        n = ProbCircuitNode(temp, prob_cache)
        cpt = get_prop(clt, root, :cpt)
        weights = [cpt[1], cpt[0]]
        n.log_thetas = log.(weights)
        push!(lin, n)
    end

    for id in topo_order
        children = Var.(outneighbors(clt, id))

        if isequal(children, [])
            compile_leaf(id)
        else
            compile_inner(id, children)
            if 0 == get_prop(clt, id, :parent)
                compile_root(id)
            end
        end

    end
    return lin
end

"check full evidence sums to 1 for prob circuits"
function prob_circuit_check(prob_circuit, data)
    EPS = 1e-7
    flow_circuit = FlowCircuit(prob_circuit, 1, Bool);
    examples_num = num_examples(data)
    N = num_features(data)

    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all,
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    data_all = XData(data_all)

    calc_prob_all = log_likelihood_per_instance(flow_circuit, data_all)
    calc_prob_all = exp.(calc_prob_all)
    sum_prob_all = sum(calc_prob_all)
    @assert sum_prob_all ≈ 1 atol = EPS;
end

"check correctness for mix of circuits"
function mix_prob_circuit_check(mix_prob, data)
    count = 1
    for m in mix_prob
        println("Checking prob circuits $count th..")
        prob_circuit_check(m, data)
        count += 1
    end
end
