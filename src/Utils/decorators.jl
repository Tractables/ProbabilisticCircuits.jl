export DecoratorCircuit, literal, origin, and_nodes, or_nodes, ⋀_nodes, ⋁_nodes,
GateType, NodeType, variable, ispositive, isnegative, origin

using LogicCircuits

"Root of the decorator circuit node hierarchy"
abstract type DecoratorCircuit <: Dag end

#####################
# functions that need to be implemented for each type of decorator circuit
#####################
"""
Get the origin node/circuit in a given decorator node/circuit
"""
function origin end

#####################
# Abstract infrastructure for decorator circuit nodes
#####################

import LogicCircuits: GateType, NodeType

"Get the gate type trait of the given `DecoratorCircuit`"
@inline GateType(instance::DecoratorCircuit) = GateType(typeof(instance))

"Map gate type traits to graph node traits"
@inline NodeType(::Type{N}) where {N<:DecoratorCircuit} = NodeType(GateType(N))

#####################
# node functions
#####################

import LogicCircuits: literal, and_nodes, or_nodes, ⋀_nodes, ⋁_nodes

"Get the logical literal in a given node"
literal(n::DecoratorCircuit)::Lit = literal(origin(n))

"Get the list of conjunction nodes in a given circuit"
⋀_nodes(c::DecoratorCircuit) = filter(is⋀gate, c)

"Get the list of And nodes in a given circuit"
@inline and_nodes(c::DecoratorCircuit) = ⋀_nodes(c)

"Get the list of disjunction nodes in a given circuit"
⋁_nodes(c::DecoratorCircuit) = filter(is⋁gate, c)

"Get the list of or nodes in a given circuit"
@inline or_nodes(c::DecoratorCircuit) = ⋁_nodes(c)

#####################
# derived node functions
#####################

import LogicCircuits: variable, ispositive, isnegative

"Get the logical variable in a given literal leaf node"
@inline variable(n::DecoratorCircuit)::Var = variable(GateType(n), origin(n))

"Get the sign of the literal leaf node"
@inline ispositive(n::DecoratorCircuit)::Bool = ispositive(GateType(n), origin(n))
@inline isnegative(n::DecoratorCircuit)::Bool = !ispositive(n)

# TODO
# istrue
# isfalse