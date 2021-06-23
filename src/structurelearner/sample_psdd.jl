using StatsFuns
using BinaryDecisionDiagrams
const BDD = BinaryDecisionDiagrams

"Samples an element from a Binomial distribution with p=0.5."
function sample_row(n::Int)::Int
    P = Float64[binompdf(n, 0.5, k) for k ∈ 1:n]
    c = 0.0
    r = rand()
    for (i, p) ∈ enumerate(P)
        c += p
        if r < c return i end
    end
    return n
end

"Samples a single combination uniformly following Robert Floyd's algorithm."
function sample_comb(n::Int, k::Int)::BitSet
    S = BitSet()
    for i ∈ n-k+1:n
        r = rand(1:i)
        if r ∉ S push!(S, r)
        else push!(S, i) end
    end
    return S
end

@inline function random_weights(n::Int)::Vector{Float64}
    W = rand(Float64, n)
    return log.(W /= sum(W))
end

@inline function get_lit(l::Int32, V::Vtree, L::Dict{Int32, StructProbLiteralNode})::StructProbLiteralNode
    if !haskey(L, l)
        node = StructProbLiteralNode(l, V)
        L[l] = node
        return node
    end
    return L[l]
end

"Options for SamplePSDD."
@enum SamplingOpts begin
    plain    = 0  # No compressions or merges.
    compress = 1  # Only compressions are allowed.
    merge    = 2  # Only merges are allowed.
    full     = 3  # Compressions and merges are allowed.
end

"""
Samples a partial partition.
"""
function sample_partition(ϕ::Diagram, Sc::BitSet, p::Real, k::Integer, ⊤_k::Integer,
        exact::Bool)::Dict{Diagram, Vector{Diagram}}
    X = intersect!(scope(ϕ), Sc)
    idem = isempty(X) || is_⊤(ϕ)
    O = shuffle!(idem ? collect(Sc) : X)
    E = Dict{Diagram, Vector{Diagram}}()
    if idem
        E[ϕ] = Vector{Diagram}()
        sample_idem_primes!(O, p, E[ϕ], ⊤_k)
    else sample_primes!(ϕ, O, E, k, exact) end
    return E
end

"""
Samples primes for the ⊤ case.
"""
function sample_idem_primes!(O::Vector{Int}, p::Real, P::Vector{Diagram}, k::Integer)
    e_count, Sc_len = 1, length(O)
    Q = Tuple{Int, Vector{Int}}[(1, Vector{Int}())]
    while !isempty(Q)
        i, V = popfirst!(Q)
        if e_count >= k || i > Sc_len
            push!(P, and(V))
            continue
        end
        x = O[i]
        c = rand() > p
        if i > 1 && c
            push!(Q, (i+1, V))
        else
            push!(Q, (i+1, push!(copy(V), x)))
            push!(Q, (i+1, push!(copy(V), -x)))
            e_count += 1
        end
    end
    nothing
end

"""
Samples primes for partial partition.
"""
function sample_primes!(ϕ::Diagram, O::Vector{Int}, E::Dict{Diagram, Vector{Diagram}}, k::Integer,
        exact::Bool)
    e_count, Sc_len = 1, length(O)
    Q = Tuple{Diagram, Int, Vector{Int}}[(ϕ, 1, Vector{Int}())]
    while !isempty(Q)
        ψ, i, V = popfirst!(Q)
        if (e_count >= k && !exact) || (i > Sc_len) || is_⊤(ψ)
            if !haskey(E, ψ) E[ψ] = Diagram[and(V)]
            else push!(E[ψ], and(V)) end
            continue
        end
        x = O[i]
        if x ∉ ψ push!(Q, (ψ, i+1, V))
        else
            α, β = ψ|x, ψ|-x
            if !is_⊥(α) push!(Q, (α, i+1, push!(copy(V), x))) end
            if !is_⊥(β) push!(Q, (β, i+1, push!(copy(V), -x))) end
            e_count += 1
        end
    end
    nothing
end

"""
Samples a PSDD from a BDD `ϕ` and vtree `V` with at most `k` elements in each disjunction node.
"""
@inline function sample_psdd(ϕ::Diagram, V::Vtree, k::Integer, D::DataFrame;
        opts::SamplingOpts = full, randomize_weights::Bool = false, pseudocount::Real = 1.0,
        fact_on_⊤::Bool = false, ⊤_k::Integer = k, p_mr::Real = 0.5, always_compress::Bool = false,
        always_merge::Bool = false, merge_branch::Real = 0.0, maxiter::Integer = 0)::StructProbCircuit
    memo = Dict{Tuple{Vtree, Diagram}, StructSumNode}()
    C = sample_psdd_r(ϕ, V, k, Dict{Int32, StructProbLiteralNode}(), randomize_weights, opts,
                      fact_on_⊤, ⊤_k, p_mr, always_compress, always_merge, memo, merge_branch > 0.0,
                      merge_branch, false, false)
    if maxiter > 0
        # Optionally grow the circuit by Strudel.
        loss(x) = heuristic_loss(x, D)
        C = struct_learn(C; primitives = [split_step], kwargs = Dict(split_step => (loss = loss,)),
                         maxiter, verbose = false)
    end
    !randomize_weights && estimate_parameters(C, D; pseudocount)
    return C
end
export sample_psdd

function sample_psdd_r(ϕ::Diagram, V::Vtree, k::Integer, leaves::Dict{Int32, StructProbLiteralNode},
        randomize_weights::Bool, opts::SamplingOpts, fact_on_⊤::Bool, ⊤_k::Integer, p_mr::Real,
        always_compress::Bool, always_merge::Bool, repeats::Dict{Tuple{Vtree, Diagram}, StructSumNode},
        merge_this::Bool, merge_branch_pr::Float64, exact::Bool, anc_exact::Bool)::StructProbCircuit
    merge_branch = merge_branch_pr > 0.0
    if merge_branch
        r_p = (V, ϕ)
        if merge_this && (merge_branch_pr > rand()) && haskey(repeats, r_p) return repeats[r_p] end
    end
    if isleaf(V)
        v, v64 = convert(Int32, V.var), convert(Int, V.var)
        if v ∈ ϕ
            if is_lit(ϕ) return get_lit(to_lit(ϕ), V, leaves) end
            if is_⊤(ϕ|v64) return get_lit(v, V, leaves) end
            if is_⊤(ϕ|-v64) return get_lit(-v, V, leaves) end
            S = StructSumNode([get_lit(v, V, leaves), get_lit(-v, V, leaves)], V)
            if merge_branch repeats[r_p] = S end
            if randomize_weights S.log_probs = random_weights(2) end
            return S
        end
        S = StructSumNode([get_lit(v, V, leaves), get_lit(-v, V, leaves)], V)
        if merge_branch repeats[r_p] = S end
        if randomize_weights S.log_probs = random_weights(2) end
        return S
    elseif fact_on_⊤ && (is_⊤(ϕ) || isempty(intersect!(scopeset(ϕ), variables(V))))
        # When ϕ ≡ ⊤ and we want to simplify the circuit, fully factorize.
        left = sample_psdd_r(⊤, V.left, k, leaves, randomize_weights, opts, fact_on_⊤, ⊤_k, p_mr,
                             always_compress, always_merge, repeats, merge_branch, merge_branch_pr,
                             false, false)
        right = sample_psdd_r(⊤, V.right, k, leaves, randomize_weights, opts, fact_on_⊤, ⊤_k, p_mr,
                              always_compress, always_merge, repeats, merge_branch,
                              merge_branch_pr, false, false)
        S = StructSumNode([StructMulNode(left, right, V)], V)
        if merge_branch repeats[r_p] = S end
        return S
    elseif length(V.variables) == 2 # when |Sc(a,b)|=2, we have only two elements: (a,ϕ|a) and (¬a,ϕ|¬a).
        prime_var = convert(Int, V.left.var)
        C = Vector{StructProbCircuit}()
        # Left element.
        left_sub_ϕ = ϕ|prime_var
        if !is_⊥(left_sub_ϕ)
            left_prime = sample_psdd_r(BDD.variable(prime_var), V.left, k, leaves,
                                       randomize_weights, opts, fact_on_⊤, ⊤_k, p_mr,
                                       always_compress, always_merge, repeats, merge_branch,
                                       merge_branch_pr, true, true)
            left_sub = sample_psdd_r(left_sub_ϕ, V.right, k, leaves, randomize_weights, opts,
                                     fact_on_⊤, ⊤_k, p_mr, always_compress, always_merge, repeats,
                                     merge_branch, merge_branch_pr, true, true)
            push!(C, StructMulNode(left_prime, left_sub, V))
        end
        right_sub_ϕ = ϕ|-prime_var
        if !is_⊥(right_sub_ϕ)
            right_prime = sample_psdd_r(BDD.variable(-prime_var), V.left, k, leaves,
                                        randomize_weights, opts, fact_on_⊤, ⊤_k, p_mr,
                                        always_compress, always_merge, repeats, merge_branch,
                                        merge_branch_pr, true, true)
            right_sub = sample_psdd_r(right_sub_ϕ, V.right, k, leaves, randomize_weights, opts,
                                      fact_on_⊤, ⊤_k, p_mr, always_compress, always_merge, repeats,
                                      merge_branch, merge_branch_pr, true, true)
            push!(C, StructMulNode(right_prime, right_sub, V))
        end
        S = StructSumNode(C, V)
        if merge_branch repeats[r_p] = S end
        if randomize_weights S.log_probs = random_weights(length(C)) end
        return S
    end
    L, R = variables(V.left), variables(V.right)
    force_exact = (exact || anc_exact) && !is_⊤(ϕ)
    E = sample_partition(ϕ, L, p_mr, k, ⊤_k, force_exact)
    C = Vector{StructProbCircuit}()
    for (s, P) ∈ E
        # Single element with sub s.
        if length(P) == 1
            # Create (p, s) and move on.
            p = first(P)
            prime_node = sample_psdd_r(p, V.left, k, leaves, randomize_weights, opts, fact_on_⊤,
                                       ⊤_k, p_mr, always_compress, always_merge, repeats,
                                       merge_branch, merge_branch_pr, true, true)
            sub_node = sample_psdd_r(s, V.right, k, leaves, randomize_weights, opts, fact_on_⊤,
                                     ⊤_k, p_mr, always_compress, always_merge, repeats,
                                     merge_branch && isempty(intersect!(scopeset(s), R)),
                                     merge_branch_pr, false, anc_exact)
            push!(C, StructMulNode(prime_node, sub_node, V))
            continue
        end
        # Else, there are length(P) elements with sub s.
        if opts == plain # no compressions nor merges.
            for p ∈ P
                prime_node = sample_psdd_r(p, V.left, k, leaves, randomize_weights, opts, fact_on_⊤,
                                           ⊤_k, p_mr, always_compress, always_merge, repeats,
                                           merge_branch, merge_branch_pr, true, true)
                sub_node = sample_psdd_r(s, V.right, k, leaves, randomize_weights, opts, fact_on_⊤,
                                         ⊤_k, p_mr, always_compress, always_merge, repeats,
                                         merge_branch && isempty(intersect!(scopeset(s), R)),
                                         merge_branch_pr, false, anc_exact)
                push!(C, StructMulNode(prime_node, sub_node, V))
            end
            continue
        end
        n = length(P)
        do_compress = Int(opts) & Int(compress) > 0
        if do_compress
            if always_compress
                K = Tuple{Diagram, Vector{Int}}[(reduce(BDD.:∨, P), Int[1])]
            else
                c = sample_row(n)
                if c == n # Compress everyone.
                    K = Tuple{Diagram, Vector{Int}}[(reduce(BDD.:∨, P), Int[1])]
                elseif c == 1 # Compress no one.
                    K = Tuple{Diagram, Vector{Int}}[(P[i], Int[i]) for i ∈ 1:n]
                else # Compress c randomly selected elements.
                    K = Vector{Tuple{Diagram, Vector{Int}}}(undef, n-c+1)
                    comb = sample_comb(n, c)
                    j, ψ = 2, ⊥
                    @inbounds for i ∈ 1:n
                        if i ∈ comb
                            ψ = BDD.:∨(ψ, P[i])
                        else
                            K[j] = (P[i], Int[j])
                            j += 1
                        end
                    end
                    K[1] = (ψ, Int[1])
                end
            end
        else
            K = Vector{Tuple{Diagram, Vector{Int}}}(undef, n)
            @inbounds for i ∈ 1:n K[i] = (P[i], Int[i]) end
        end
        m = length(K)
        if (Int(opts) & Int(merge)) > 0 # then merge.
            c = sample_row(m)
            if always_merge || c == m
                M = Vector{Int}(undef, m)
                @inbounds for i ∈ 2:m
                    M[i] = i
                    K[i] = (K[i][1], Int[])
                end
                M[1] = 1
                K[1] = (K[1][1], M)
            elseif c != 1 # if c == 1, no need to merge.
                comb = sample_comb(m, c)
                M = Vector{Int}(undef, c)
                j = 1
                for i ∈ 1:m
                    if i ∈ comb
                        M[j] = i
                        if j == 1 K[i] = (K[i][1], M)
                        else K[i] = (K[i][1], Int[]) end
                        j += 1
                    end
                end
            end
        end
        # Create subs.
        subs = Vector{StructProbCircuit}(undef, m)
        for (i, c) ∈ enumerate(K)
            M = c[2]
            if !isempty(M)
                sub_node = sample_psdd_r(s, V.right, k, leaves, randomize_weights, opts, fact_on_⊤,
                                         ⊤_k, p_mr, always_compress, always_merge, repeats,
                                         merge_branch && isempty(intersect!(scopeset(s), R)),
                                         merge_branch_pr, false, anc_exact)
                for j ∈ M subs[j] = sub_node end
            end
        end
        for (i, c) ∈ enumerate(K)
            α, M = c[1], c[2]
            prime_node = sample_psdd_r(α, V.left, k, leaves, randomize_weights, opts, fact_on_⊤,
                                       ⊤_k, p_mr, always_compress, always_merge, repeats,
                                       merge_branch, merge_branch_pr, true, true)
            prod_node = StructMulNode(prime_node, subs[i], V)
            push!(C, prod_node)
        end
    end
    S = StructSumNode(C, V)
    if merge_branch repeats[r_p] = S end
    if randomize_weights S.log_probs = random_weights(length(C)) end
    return S
end
