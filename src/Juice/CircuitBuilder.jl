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

    "compile inner node, 1 inner varibale to 2 leaf nodes, 2 * num_children disjunction nodes and 2 conjunction nodes"
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


"simple test code to parse a chow-liu tree"
function test_parse_tree(filename::String)
    f = open(filename)
    n = parse(Int32,readline(f))
    clt = MetaDiGraph(n)
    root, prob = split(readline(f), " ")
    root, prob = parse(Int32, root), parse(Float64, prob)
    set_prop!(clt, root, :parent, 0)
    set_prop!(clt, root, :cpt, Dict(1=>prob,0=>1-prob))
    for i = 1:n-1
        dst, src, prob1, prob0 = split(readline(f), " ")
        dst, src, prob1, prob0 = parse(Int32, dst), parse(Int32, src), parse(Float64, prob1), parse(Float64, prob0)
        add_edge!(clt, src,dst)
        set_prop!(clt, dst, :parent, src)
        set_prop!(clt, dst, :cpt, Dict((1,1)=>prob1, (0,1)=>1-prob1, (1,0)=>prob0, (0,0)=>1-prob0))
    end
    return clt
end
