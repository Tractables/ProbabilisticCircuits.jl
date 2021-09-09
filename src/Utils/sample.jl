export sample_vtree

using Random
using LogicCircuits: Vtree, PlainVtreeLeafNode

#####################
# Sampling functions
#####################

"""Split `X` into two partitions `A` and `B`, where `A` is a Bernoulli sample of each element in
`X` with probability `p` and `B=X∖A`. Guarantees at least one element in `A` and `B`."""
function bernoulli_partition(X::Vector{Int}, p::Float64)::Tuple{Vector{Int}, Vector{Int}}
    n = length(X)
    a = rand(1:n)
    b = ((rand(0:n-2)+a)%n)+1
    A, B = Int[X[a]], Int[X[b]]
    for (i, x) ∈ enumerate(X)
        if i == a continue end
        if i == b continue end
        if rand() > p push!(B, x)
        else push!(A, x) end
    end
    return A, B
end

"Samples a Vtree with a right bias of `p`. If `p<0`, then uniformly sample vtrees."
function sample_vtree(n::Int, p::Float64)::Vtree
    passdown(x::Int)::Vtree = PlainVtreeLeafNode(x)
    function passdown(X::Vector{Int})::Vtree
        R, L = bernoulli_partition(X, p)
        return Vtree(passdown(length(L) == 1 ? L[1] : L), passdown(length(R) == 1 ? R[1] : R))
    end
    return p < 0 ? Vtree(n, :random) : passdown(shuffle!(collect(1:n)))
end
