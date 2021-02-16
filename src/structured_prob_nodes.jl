export ProbCircuit, StructProbCircuit, StructProbLeafNode, StructProbInnerNode,
    StructProbLiteralNode, StructMulNode, StructSumNode, check_parameter_integrity

#####################
# Prob circuits that are structured,
# meaning that each conjunction is associated with a vtree node.
#####################

"Root of the plain structure probabilistic circuit node hierarchy"
abstract type StructProbCircuit <: ProbCircuit end

"A plain structured probabilistic leaf node"
abstract type StructProbLeafNode <: StructProbCircuit end

"A plain structured probabilistic inner node"
abstract type StructProbInnerNode <: StructProbCircuit end

"A plain structured probabilistic literal leaf node, representing the positive or negative literal of its variable"
mutable struct StructProbLiteralNode <: StructProbLeafNode
    literal::Lit
    vtree::Vtree
    data
    counter::UInt32
    StructProbLiteralNode(l,v) = begin
        @assert lit2var(l) ∈ v 
        new(l, v, nothing, 0)
    end
end

"A plain structured probabilistic conjunction node"
mutable struct StructMulNode <: StructProbInnerNode
    prime::StructProbCircuit
    sub::StructProbCircuit
    vtree::Vtree
    data
    counter::UInt32
    StructMulNode(p,s,v) = begin
        @assert isinner(v) "Structured conjunctions must respect inner vtree node"
        @assert varsubset_left(vtree(p),v) "$p does not go left in $v"
        @assert varsubset_right(vtree(s),v) "$s does not go right in $v"
        new(p,s, v, nothing, 0)
    end
end

"A plain structured probabilistic disjunction node"
mutable struct StructSumNode <: StructProbInnerNode
    children::Vector{StructProbCircuit}
    log_probs::Vector{Float64}
    vtree::Vtree # could be leaf or inner
    data
    counter::UInt32
    StructSumNode(c, v) = 
        new(c, log.(ones(Float64, length(c)) / length(c)), v, nothing, 0)
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:StructProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:StructMulNode}) = ⋀Gate()
@inline GateType(::Type{<:StructSumNode}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children, vtree, vtree_safe, respects_vtree # make available for extension
@inline children(n::StructSumNode) = n.children
@inline children(n::StructMulNode) = [n.prime,n.sub]

"Get the vtree corresponding to the argument, or nothing if the node has no vtree"
@inline vtree(n::StructProbCircuit) = n.vtree
@inline vtree_safe(n::StructProbInnerNode) = vtree(n)
@inline vtree_safe(n::StructProbLiteralNode) = vtree(n)

# ProbCircuit has a default argument for respects: its root's vtree
respects_vtree(circuit::StructProbCircuit) = 
    respects_vtree(circuit, vtree(circuit))

@inline num_parameters_node(n::StructSumNode) = num_children(n)

#####################
# constructors and compilation
#####################

multiply(arguments::Vector{<:StructProbCircuit};
        reuse=nothing, use_vtree=nothing) =
        multiply(arguments...; reuse, use_vtree)

function multiply(a1::StructProbCircuit,  
                 a2::StructProbCircuit;
                 reuse=nothing, use_vtree=nothing) 
    reuse isa StructMulNode && reuse.prime == a1 && reuse.sub == a2 && return reuse
    !(use_vtree isa Vtree) && (reuse isa StructProbCircuit) &&  (use_vtree = reuse.vtree)
    !(use_vtree isa Vtree) && (use_vtree = find_inode(vtree_safe(a1), vtree_safe(a2)))
    return StructMulNode(a1, a2, use_vtree)
end

function summate(arguments::Vector{<:StructProbCircuit};
                 reuse=nothing, use_vtree=nothing)
    @assert length(arguments) > 0
    reuse isa StructSumNode && reuse.children == arguments && return reuse
    !(use_vtree isa Vtree) && (reuse isa StructProbCircuit) &&  (use_vtree = reuse.vtree)
    !(use_vtree isa Vtree) && (use_vtree = mapreduce(vtree_safe, lca, arguments))
    return StructSumNode(arguments, use_vtree)
end

# claim `StructProbCircuit` as the default `ProbCircuit` implementation that has a vtree

compile(::Type{ProbCircuit}, a1::Union{Vtree, StructLogicCircuit}, args...) =
    compile(StructProbCircuit, a1, args...)

compile(n::StructProbCircuit, args...) = 
    compile(typeof(n), root(vtree(n)), args...)

compile(::Type{<:StructProbCircuit}, c::StructLogicCircuit) =
    compile(StructProbCircuit, root(vtree(c)), c)

compile(::Type{<:StructLogicCircuit}, c::StructProbCircuit) =
    compile(StructLogicCircuit, root(vtree(c)), c)

compile(::Type{<:StructProbCircuit}, ::Vtree, ::Bool) =
    error("Probabilistic circuits do not have constant leafs.")

compile(::Type{<:StructProbCircuit}, vtree::Vtree, l::Lit) =
    StructProbLiteralNode(l,find_leaf(lit2var(l),vtree))

function compile(::Type{<:StructProbCircuit}, vtree::Vtree, circuit::LogicCircuit)
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = compile(StructProbCircuit, vtree, literal(n))
    f_a(n, cns) = multiply(cns...) # note: this will use the LCA as vtree node
    f_o(n, cns) = summate(cns) # note: this will use the LCA as vtree node
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, StructProbCircuit)
end

function Base.convert(::Type{<:StructProbCircuit}, sdd::Sdd)::StructProbCircuit
    L = Dict{Int32, StructProbLiteralNode}()
    visited = Dict{Sdd, StructProbCircuit}()
    Sc_sdd = variables_by_node(sdd)
    ⊤_node = SddTrueNode(false, nothing)
    function get_lit(l::Int32, V::Vtree, L::Dict{Int32, StructProbLiteralNode})::StructProbLiteralNode
      if !haskey(L, l)
        node = StructProbLiteralNode(l, V)
        L[l] = node
        return node
      end
      return L[l]
    end
    function passdown(S::Sdd, V::Vtree, ignore::Bool = false)::StructProbCircuit
        if !ignore && haskey(visited, S) return visited[S] end
        if S isa SddTrueNode
            Sc = variables(V)
            if length(Sc) == 1
                l = convert(Int32, first(Sc))
                return StructSumNode([get_lit(l, V, L), get_lit(-l, V, L)], V)
            end
            # Fully factorize.
            left, right = passdown(S, V.left, true), passdown(S, V.right, true)
            return StructSumNode([StructMulNode(left, right, V)], V)
        elseif S isa SddLiteralNode
            l = S.literal
            Sc = variables(V)
            if length(Sc) == 1 return get_lit(l, V, L) end
            v = abs(l)
            left = passdown(v ∈ variables(V.left) ? S : ⊤_node, V.left, true)
            right = passdown(v ∈ variables(V.right) ? S : ⊤_node, V.right, true)
            return StructSumNode([StructMulNode(left, right, V)], V)
        end
        # Else, disjunction node.
        ch = Vector{StructProbCircuit}()
        for c ∈ S.children
            if c.sub isa SddFalseNode continue end
            if haskey(visited, c)
                push!(ch, visited[c])
                continue
            end
            p = passdown(c.prime, V.left)
            # Corner case for when SDD is not smooth on the left. Add missing variables.
            if !(V.left isa PlainVtreeLeafNode)
                if V.left.left isa PlainVtreeLeafNode && V.left.left.var ∉ variables(vtree(p))
                    p = StructSumNode([StructMulNode(passdown(⊤_node, V.left.left, true), p, V.left)], V.left)
                end; if V.left.right isa PlainVtreeLeafNode && V.left.right.var ∉ variables(vtree(p))
                    p = StructSumNode([StructMulNode(p, passdown(⊤_node, V.left.right, true), V.left)], V.left)
                end
            end
            s = passdown(c.sub, V.right)
            # Corner case for when SDD is not smooth on the right. Add missing variables.
            if !(V.right isa PlainVtreeLeafNode)
                if V.right.left isa PlainVtreeLeafNode && V.right.left.var ∉ variables(vtree(s))
                    s = StructSumNode([StructMulNode(passdown(⊤_node, V.right.left, true), s, V.right)], V.right)
                end; if V.right.right isa PlainVtreeLeafNode && V.right.right.var ∉ variables(vtree(s))
                    s = StructSumNode([StructMulNode(s, passdown(⊤_node, V.right.right, true), V.right)], V.right)
                end
            end
            e = StructMulNode(p, s, V)
            visited[c] = e
            push!(ch, e)
        end
        sum = StructSumNode(ch, V)
        visited[S] = sum
        return sum
    end
    return passdown(sdd, Vtree(sdd.vtree))
end

function fully_factorized_circuit(::Type{<:ProbCircuit}, vtree::Vtree)
    ff_logic_circuit = fully_factorized_circuit(PlainStructLogicCircuit, vtree)
    compile(StructProbCircuit, vtree, ff_logic_circuit)
end
