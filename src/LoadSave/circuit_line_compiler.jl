#####################
# Compilers to ProbabilisticCircuits data structures starting from already parsed line objects
#####################

# reuse some internal infrastructure of LogicCircuits' LoadSave module
using LogicCircuits.LoadSave: CircuitFormatLines, CircuitFormatLine, constant,
VtreeFormatLines, CircuitHeaderLine, UnweightedLiteralLine, WeightedLiteralLine, 
DecisionLine, LCElement, BiasLine, WeightedNamedConstantLine, PSDDElement, 
CircuitCommentLine, ID, compile_smooth_struct_logical_m, compile_smooth_logical_m

"""
Compile lines into a probabilistic circuit
"""
function compile_prob(lines::CircuitFormatLines)::ProbCircuit
    # first compile a logical circuit
    logic_circuit, id2lognode = compile_smooth_logical_m(lines)
    decorate_prob(lines, logic_circuit, id2lognode)
end

"""
Compile lines into a logistic circuit.
"""
# TODO
# function compile_logistic(lines::CircuitFormatLines, classes::Int)::LogisticΔ
#     # first compile a logical circuit
#     logic_circuit, id2lognode = compile_smooth_logical_m(lines)
#     decorate_logistic(lines, logic_circuit, classes, id2lognode)
# end

"""
Compile circuit and vtree lines into a structured probabilistic circuit (one whose logical circuit origin is structured).
"""
function compile_struct_prob(circuit_lines::CircuitFormatLines, vtree_lines::VtreeFormatLines)
    logic_circuit, vtree, id2lognode, id2vtree = compile_smooth_struct_logical_m(circuit_lines, vtree_lines)
    prob_circuit = decorate_prob(circuit_lines, logic_circuit, id2lognode)
    return prob_circuit, vtree
end

function decorate_prob(lines::CircuitFormatLines, logic_circuit::LogicCircuit, id2lognode::Dict{ID,<:LogicCircuit})::ProbCircuit
    
    # set up cache mapping logical circuit nodes to their probabilistic decorator
    prob_circuit = ProbCircuit(logic_circuit)
    lognode2probnode = Dict{LogicCircuit, ProbCircuit}()
    foreach(pn -> (lognode2probnode[origin(pn)] = pn), prob_circuit) 

    # map from line node ids to probabilistic circuit nodes
    id2probnode(id) = lognode2probnode[id2lognode[id]]

    root = nothing

    # go through lines again and update the probabilistic circuit node parameters

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln into probabilistic circuit is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine,UnweightedLiteralLine})
        # do nothing
    end
    function compile(ln::WeightedNamedConstantLine)
        @assert constant(ln) == true
        root = id2probnode(ln.node_id)::Prob⋁Node
        root.log_thetas .= [ln.weight, log1p(-exp(ln.weight))]
    end
    function compile(ln::DecisionLine{<:PSDDElement})
        root = id2probnode(ln.node_id)::Prob⋁Node
        root.log_thetas .= [x.weight for x in ln.elements]
    end

    foreach(compile, lines)

    root
end

# TODO
# function decorate_logistic(lines::CircuitFormatLines, logic_circuit::LogicΔ, 
#                             classes::Int, id2lognode::Dict{ID,<:LogicCircuit})::LogisticΔ
                        
#     # set up cache mapping logical circuit nodes to their logistic decorator
#     log2logistic = LogisticCache()
#     # build a corresponding probabilistic circuit
#     logistic_circuit = LogisticΔ(logic_circuit, classes, log2logistic)
#     # map from line node ids to probabilistic circuit nodes
#     id2logisticnode(id) = log2logistic[id2lognode[id]]

#     # go through lines again and update the probabilistic circuit node parameters

#     function compile(ln::CircuitFormatLine)
#         error("Compilation of line $ln into logistic circuit is not supported")
#     end
#     function compile(::Union{CircuitHeaderLine,CircuitCommentLine,UnweightedLiteralLine})
#         # do nothing
#     end

#     function compile(ln::CircuitHeaderLine)
#         # do nothing
#     end

#     function compile(ln::WeightedLiteralLine)
#         node = id2logisticnode(ln.node_id)::Logistic⋁
#         node.thetas[1, :] .= ln.weights
#     end

#     function compile(ln::DecisionLine{<:LCElement})
#         node = id2logisticnode(ln.node_id)::Logistic⋁
#         for (ind, elem) in enumerate(ln.elements)
#             node.thetas[ind, :] .= elem.weights
#         end
#     end

#     function compile(ln::BiasLine)
#         node = id2logisticnode(ln.node_id)::Logistic⋁
#         # @assert length(node.thetas) == 1
#         node.thetas[1,:] .= ln.weights
#     end

#     for ln in lines
#         compile(ln)
#     end

#     logistic_circuit
# end