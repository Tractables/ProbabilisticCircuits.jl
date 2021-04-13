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
    StructProbLiteralNode(l,v) = begin
        @assert lit2var(l) ∈ v 
        new(l, v)
    end
end

"A plain structured probabilistic conjunction node"
mutable struct StructMulNode <: StructProbInnerNode
    prime::StructProbCircuit
    sub::StructProbCircuit
    vtree::Vtree
    StructMulNode(p,s,v) = begin
        @assert isinner(v) "Structured conjunctions must respect inner vtree node"
        @assert varsubset_left(vtree(p),v) "$p does not go left in $v"
        @assert varsubset_right(vtree(s),v) "$s does not go right in $v"
        new(p,s, v)
    end
end

"A plain structured probabilistic disjunction node"
mutable struct StructSumNode <: StructProbInnerNode
    children::Vector{StructProbCircuit}
    log_probs::Vector{Float64}
    vtree::Vtree # could be leaf or inner
    StructSumNode(c, v) = 
        new(c, log.(ones(Float64, length(c)) / length(c)), v)
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

function compile(::Type{<:StructProbCircuit}, sdd::Sdd)::StructProbCircuit
    lc = LogicCircuit(sdd)
    plc = propagate_constants(lc, remove_unary=true)
    structplc = compile(StructLogicCircuit, vtree(sdd), plc)
    sstructplc = smooth(structplc)
    compile(StructProbCircuit, sstructplc)
end

function fully_factorized_circuit(::Type{<:ProbCircuit}, vtree::Vtree)
    ff_logic_circuit = fully_factorized_circuit(PlainStructLogicCircuit, vtree)
    compile(StructProbCircuit, vtree, ff_logic_circuit)
end
