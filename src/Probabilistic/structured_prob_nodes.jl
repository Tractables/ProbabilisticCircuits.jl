export ProbCircuit, StructProbCircuit,  StructProbLeafNode, StructProbInnerNode,
    StructProbLiteralNode, StructProb⋀Node, StructProb⋁Node, check_parameter_integrity

#####################
# Prob circuits that are structured,
# meaning that each conjunction is associated with a vtree node.
#####################

"Root of the plain structure probabilistic circuit node hierarchy"
abstract type StructProbCircuit <: StructLogicCircuit end

"Root of the probabilistic circuit node hierarchy"
const ProbCircuit = Union{StructProbCircuit, PlainProbCircuit}

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
mutable struct StructProb⋀Node <: StructProbInnerNode
    prime::StructProbCircuit
    sub::StructProbCircuit
    vtree::Vtree
    data
    counter::UInt32
    StructProb⋀Node(p,s,v) = begin
        @assert isinner(v) "Structured conjunctions must respect inner vtree node"
        @assert varsubset_left(vtree(p),v) "$p does not go left in $v"
        @assert varsubset_right(vtree(s),v) "$s does not go right in $v"
        new(p,s, v, nothing, 0)
    end
end

"A plain structured probabilistic disjunction node"
mutable struct StructProb⋁Node <: StructProbInnerNode
    children::Vector{<:StructProbCircuit}
    log_thetas::Vector{Float64}
    vtree::Vtree # could be leaf or inner
    data
    counter::UInt32
    StructProb⋁Node(c, v) = new(c, init_array(Float64, length(c)), v, nothing, 0)
end

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
@inline GateType(::Type{<:StructProbLiteralNode}) = LiteralGate()
@inline GateType(::Type{<:StructProb⋀Node}) = ⋀Gate()
@inline GateType(::Type{<:StructProb⋁Node}) = ⋁Gate()

#####################
# methods
#####################

import LogicCircuits: children # make available for extension
@inline children(n::StructProb⋁Node) = n.children
@inline children(n::StructProb⋀Node) = [n.prime,n.sub]

import ..Utils: num_parameters
@inline num_parameters(c::StructProbCircuit) = sum(n -> num_children(n), ⋁_nodes(c))

#####################
# constructors and conversions
#####################

function StructProbCircuit(circuit::PlainStructLogicCircuit)::StructProbCircuit
    f_con(n) = error("Cannot construct a struct probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = StructProbLiteralNode(literal(n), vtree(n))
    f_a(n, cn) = begin
        @assert length(cn)==2
        StructProb⋀Node(cn[1], cn[2], vtree(n))
    end
    f_o(n, cn) = StructProb⋁Node(cn, vtree(n))
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, StructProbCircuit)
end 

function PlainStructLogicCircuit(circuit::StructProbCircuit)::PlainStructLogicCircuit
    f_con(n) = error("Cannot construct a struct probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = PlainStructLiteralNode(literal(n), vtree(n))
    f_a(n, cn) = begin
        @assert length(cn)==2
        PlainStruct⋀Node(cn[1], cn[2], vtree(n))
    end
    f_o(n, cn) = PlainStruct⋁Node(cn, vtree(n))
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, PlainStructLogicCircuit)
end

@inline ProbCircuit(circuit::PlainStructLogicCircuit) = StructProbCircuit(circuit)

@inline ProbCircuit(circuit::PlainLogicCircuit) = PlainProbCircuit(circuit)

import LogicCircuits: conjoin, disjoin, compile, vtree, vtree_safe # make available for extension

"Get the vtree corresponding to the argument, or nothing if the node has no vtree"
@inline vtree(n::StructProbCircuit) = n.vtree
@inline vtree_safe(n::StructProbInnerNode) = vtree(n)
@inline vtree_safe(n::StructProbLiteralNode) = vtree(n)

conjoin(arguments::Vector{<:StructProbCircuit};
        reuse=nothing, use_vtree=nothing) =
        conjoin(arguments...; reuse, use_vtree)

function conjoin(a1::StructProbCircuit,  
                 a2::StructProbCircuit;
                 reuse=nothing, use_vtree=nothing) 
    reuse isa StructProb⋀Node && reuse.prime == a1 && reuse.sub == a2 && return reuse
    !(use_vtree isa Vtree) && (reuse isa StructProbCircuit) &&  (use_vtree = reuse.vtree)
    # if isconstantgate(a1) && isconstantgate(a2) && !(use_vtree isa Vtree)
    #     # constant nodes don't have a vtree: resolve to a constant
    #     return StructProbCircuit(istrue(a1) && istrue(a2))
    # end
    !(use_vtree isa Vtree) && (use_vtree = find_inode(vtree_safe(a1), vtree_safe(a2)))
    return StructProb⋀Node(a1, a2, use_vtree)
end

# ProbCircuit has a default argument for respects: its root's vtree
respects_vtree(circuit::ProbCircuit) = 
    respects_vtree(circuit, vtree(circuit))

@inline disjoin(xs::StructProbCircuit...) = disjoin(collect(xs))

function disjoin(arguments::Vector{<:StructProbCircuit};
                 reuse=nothing, use_vtree=nothing)
    @assert length(arguments) > 0
    reuse isa StructProb⋁Node && reuse.children == arguments && return reuse
    !(use_vtree isa Vtree) && (reuse isa StructProbCircuit) &&  (use_vtree = reuse.vtree)
    # if all(isconstantgate, arguments) && !(use_vtree isa Vtree)
    #     # constant nodes don't have a vtree: resolve to a constant
    #     return StructProbCircuit(any(constant, arguments))
    # end
    !(use_vtree isa Vtree) && (use_vtree = mapreduce(vtree_safe, lca, arguments))
    return StructProb⋁Node(arguments, use_vtree)
end

# Syntactic sugar for compile with a vtree
(t::Tuple{<:Type,<:Vtree})(arg) = compile(t[1], t[2], arg)
(t::Tuple{<:Vtree,<:Type})(arg) = compile(t[2], t[1], arg)

# claim `StructProbCircuit` as the default `ProbCircuit` implementation that has a vtree
compile(::Type{ProbCircuit}, args...) =
    compile(StructProbCircuit, args...)

compile(vtree::Vtree, arg) = 
    compile(ProbCircuit,vtree,arg)

compile(::Type{<:StructProbCircuit}, ::Vtree, b::Bool) =
    compile(StructProbCircuit, b)

"The unique splain tructured logical false constant" # act as a place holder in `condition`
const structfalse = PlainStructFalseNode(nothing, 0)

compile(::Type{<:StructProbCircuit}, b::Bool) =
    b ? structtrue : structfalse



compile(::Type{<:StructProbCircuit}, vtree::Vtree, l::Lit) =
    PlainStructLiteralNode(l,find_leaf(lit2var(l),vtree))


function compile(::Type{<:StructProbCircuit}, vtree::Vtree, circuit::StructProbCircuit)
    f_con(n) = error("ProbCircuit does not have a constant node")
    f_lit(n) = compile(StructProbCircuit, vtree, literal(n))
    f_a(n, cns) = conjoin(cns...) # note: this will use the LCA as vtree node
    f_o(n, cns) = disjoin(cns) # note: this will use the LCA as vtree node
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, StructProbCircuit)
end


function check_parameter_integrity(circuit::ProbCircuit)
    for node in or_nodes(circuit)
        @assert all(θ -> !isnan(θ), node.log_thetas) "There is a NaN in one of the log_thetas"
    end
    true
end
