using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs

"convert literal+/- to probability value 0/1"
@inline lit2value(l::Lit)::Int = (l > 0 ? 1 : 0)


"build probability circuits from Baysian tree "
function compile_prob_circuit_from_clt(clt::MetaDiGraph)::Vector{ProbCircuitNode}
    topo_order = Var.(reverse(topological_sort_by_dfs(clt))) #order to parse the node
    lin = Vector{ProbCircuitNode}()
    node_cache = Dict{Lit, LogicalCircuitNode}()
    prob_cache = ProbCache()

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

    "compile inner disjunction node into circuits"
    function compile_⋁inner(ln::Lit, children::Vector{Var})::⋁Node
        #build logical ciruits
        signed_children = vec([var2lit.(children) -var2lit.(children)]')
        temp = ⋁Node([node_cache[s] for s in signed_children])
        n = ProbCircuitNode(temp, prob_cache)

        #calculate weights for each children
        cpts = [get_prop(clt, c, :cpt) for c in children]
        v = lit2value(ln)
        weights = vec([[c[(1,v)] for c in cpts] [c[(0,v)] for c in cpts]]')
        n.log_thetas = log.(weights)
        push!(lin, n)
        return temp
    end

    "compile inner conjunction node into circuits, left node is indicator, right node is disjunction children nodes"
    function compile_⋀inner(left::Lit, right::⋁Node)
        leaf = (left > 0 ? PosLeafNode(left) : NegLeafNode(-left))
        prob_leaf = ProbCircuitNode(leaf, prob_cache)
        temp = ⋀Node([leaf,right])
        n = ProbCircuitNode(temp, prob_cache)
        node_cache[left] = temp
        push!(lin, prob_leaf)
        push!(lin, n)
    end

    "compile variable into 4 inner nodes"
    function compile_inner(ln::Var, children::Vector{Var})
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
function test_parse_tree()
    f = open(pwd()*"/src/Juice/test.txt")
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

"
4
3 0.5
1 2 0.4 0.5
2 3 0.1 0.5
4 3 0.3 0.3
"
