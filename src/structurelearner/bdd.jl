export learn_bdd

"Returns an increasing sorted right-linear vtree."
function bdd_vtree!(V::AbstractVector{<:Integer})::Vtree
    passdown(x::Integer)::Vtree = PlainVtreeLeafNode(x)
    passdown(X::AbstractVector{<:Integer})::Vtree = Vtree(passdown(popfirst!(X)),
                                                          passdown(length(X) == 1 ? X[1] : X))
    return passdown(sort!(V))
end

"Returns a fully factorized circuit reusing leaf nodes from `L`."
function factorize_reuse(X::Vector{Int}, L::Dict{Int32, StructProbLiteralNode})::StructProbCircuit
    U = Vtree([PlainVtreeLeafNode(x) for x ∈ X], :random)
    function passdown(V::Vtree)::StructProbCircuit
        if isleaf(V)
            v = convert(Int32, V.var)
            return StructSumNode([get_lit(-v, V, L), get_lit(v, V, L)], V)
        end
        left = passdown(V.left)
        right = passdown(V.right)
        return StructSumNode([StructMulNode(left, right, V)], V)
    end
    return passdown(U)
end

"Returns a structured probabilistic circuit compiled from a binary decision diagram."
function generate_bdd(ϕ::Diagram, n::Integer)::StructProbCircuit
    Sc = BDD.scope(ϕ)
    X = setdiff!(collect(1:n), Sc)
    U = bdd_vtree!(Sc)
    L = Dict{Int32, StructProbLiteralNode}()
    visited = Dict{UInt64, StructSumNode}()
    function passdown(V::Vtree, α::Diagram)::StructProbCircuit
        sh = hash((BDD.shallowhash(α), V))
        if haskey(visited, sh) return visited[sh] end
        if BDD.is_⊤(α)
            if isleaf(V)
                v = convert(Int32, V.var)
                return StructSumNode([get_lit(-v, V, L), get_lit(v, V, L)], V)
            end
            v = convert(Int32, V.left.var)
            sub = passdown(V.right, α)
            return StructSumNode([StructMulNode(get_lit(-v, V.left, L), sub, V),
                                  StructMulNode(get_lit(v, V.left, L), sub, V)], V)
        end
        v = convert(Int32, α.index)
        if isleaf(V) && BDD.is_lit(α) return get_lit(BDD.to_lit(α), V, L) end
        C = StructProbCircuit[]
        if V.left.var != v
            sub = passdown(V.right, α)
            v = convert(Int32, V.left.var)
            push!(C, StructMulNode(get_lit(-v, V.left, L), sub, V))
            push!(C, StructMulNode(get_lit(v, V.left, L), sub, V))
        else
            if !BDD.is_⊥(α.low) push!(C, StructMulNode(get_lit(-v, V.left, L), passdown(V.right, α.low), V)) end
            if !BDD.is_⊥(α.high) push!(C, StructMulNode(get_lit(v, V.left, L), passdown(V.right, α.high), V)) end
        end
        s = StructSumNode(C, V)
        visited[sh] = s
        return s
    end
    f = passdown(U, ϕ)
    if isempty(X) return f end
    p = factorize_reuse(X, L)
    if isempty(Sc) return p end
    V = Vtree(U, p.vtree)
    pc = StructSumNode([StructMulNode(f, p, V)], V)
    return pc
end

"Learns a structured probabilistic circuit consistent with a binary decision diagram `ϕ`."
function learn_bdd(ϕ::Diagram, D::DataFrame; pseudocount::Real)::StructProbCircuit
    pc = generate_bdd(ϕ, ncol(D))
    estimate_parameters(pc, D; pseudocount)
    return pc
end
