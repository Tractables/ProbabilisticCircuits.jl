export compile_factor_graph, fg_to_cnf, wmc_chavira, get_varprob

"""
    wmc_chavira(root::LogicCircuit; varprob::Function)::Float

Compute weighted model count in the context of a chavira encoding (default negative literals to 1)
Probability of each variable is given by `varprob` Function which defauls to 1/2 for every variable.
"""
function wmc_chavira(root::LogicCircuit, 
                  varprob::Function = v -> 0.5)::Float32
    f_con(n) = istrue(n) ? 1.0 : 0.0
    f_lit(n) = ispositive(n) ? varprob(variable(n)) : 1.0
    f_a(n, call) = mapreduce(call, *, children(n))
    f_o(n, call) = mapreduce(call, +, children(n))
    foldup(root, f_con, f_lit, f_a, f_o, Float64)
end

function get_varprob(fg::FactorGraph, var_lits, fac_lits)
    d = Dict()
    # Just give ever variable indicator literal weight 1 for now
    for kv in collect(var_lits)
        for literal in kv[2]  
              d[variable(literal)] = 1.0
        end
    end

    for fac in fg.facs
        for k in keys(fac.factor)
            lit = fac_lits[fac][k]
            weight = fac.factor[k]
            d[variable(lit)] = weight
        end
    end

    x -> d[x]
end


function bijection(a, b)
    (!a | b) & (a | !b)
end

function biject_cnf(a, b)
    # Here we assume that A is a list of literals and b is a literal
    c = b | disjoin(map(x -> !x, a))
    c & conjoin(map(x -> !b | x, a))
end

function fg_to_cnf(fg::FactorGraph)
    curr = 0
    # So we can get ids for each
    function nextSDDVar()
        curr += 1
        return curr
    end

    # Use a balanced vtree for now
    total_lits = sum(v.dim for v in fg.vars) + sum(length(keys(f.factor)) for f in fg.facs)
    vtr = PlainVtree(total_lits, :balanced)
    mgr = SddMgr(vtr) 

    # Create lits for varnodes
    var_lits = Dict(v => [PlainLiteralNode(nextSDDVar()) for i = 1:v.dim] for v in fg.vars)
    println(var_lits)

    # lits for factor nodes
    fac_lits = Dict(f => Dict(k => PlainLiteralNode(nextSDDVar()) for k in keys(f.factor)) for f in fg.facs)
    println(fac_lits)

    @assert total_lits == curr

    # Start CNF with true
    c = PlainTrueNode()

    # Compile indicator CNFs
    for varnode in fg.vars
        for i = 2:varnode.dim
            for j = 1:i-1
                c = c & (!var_lits[varnode][i] | !var_lits[varnode][j])
            end
        end
    end

    # Compile factor CNFs
    for facnode in fg.facs
        for config in keys(facnode.factor)
            fc = [var_lits[varnode][assignment] for (assignment, varnode) in zip(config, facnode.neighbs)]
            c = c & biject_cnf(fc, fac_lits[facnode][config])
        end
    end
    compile(mgr, c)
end

function compile_factor_graph(fg::FactorGraph)
    curr = 0
    # So we can get ids for each
    function nextSDDVar()
        curr += 1
        return curr
    end

    # Use a balanced vtree for now
    total_lits = sum(v.dim for v in fg.vars) + sum(length(keys(f.factor)) for f in fg.facs)
    vtr = PlainVtree(total_lits, :balanced)
    mgr = SddMgr(vtr) 

    # Create lits for varnodes
    var_lits = Dict(v => [compile(mgr, Lit(nextSDDVar())) for i = 1:v.dim] for v in fg.vars)
    println(var_lits)

    # lits for factor nodes
    fac_lits = Dict(f => Dict(k => compile(mgr, Lit(nextSDDVar())) for k in keys(f.factor)) for f in fg.facs)
    println(fac_lits)

    @assert total_lits == curr

    # Begin compilation
    c = compile(mgr, true)

    # Compile indicator CNFs
    for varnode in fg.vars
        for i = 2:varnode.dim
            for j = 1:i-1
                c = c & (!var_lits[varnode][i] | !var_lits[varnode][j])
                c = c & (var_lits[varnode][i] | var_lits[varnode][j])
            end
        end
    end

    # Compile factor CNFs
    for facnode in fg.facs
        for config in keys(facnode.factor)
            fc = compile(mgr, true)
            for (assignment, varnode) in zip(config, facnode.neighbs)
                fc = fc & var_lits[varnode][assignment]
            end
            c = c & bijection(fc, fac_lits[facnode][config])
        end
    end
    c, var_lits, fac_lits
end