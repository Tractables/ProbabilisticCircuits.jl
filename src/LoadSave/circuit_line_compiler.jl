#####################
# Compilers to ProbabilisticCircuits data structures starting from already parsed line objects
#####################

# reuse some internal infrastructure of LogicCircuits' LoadSave module
using LogicCircuits.LoadSave: CircuitFormatLines, CircuitFormatLine, lnconstant,
VtreeFormatLines, CircuitHeaderLine, UnweightedLiteralLine, WeightedLiteralLine, 
DecisionLine, LCElement, BiasLine, WeightedNamedConstantLine, PSDDElement, 
CircuitCommentLine, ID, compile_smooth_struct_logical_m, compile_smooth_logical_m

"""
Compile lines into a probabilistic circuit
"""
function compile_prob(lines::CircuitFormatLines)::ProbCircuit
    # first compile a logic circuit
    logic_circuit, id2lognode = compile_smooth_logical_m(lines)
    decorate_prob(lines, logic_circuit, id2lognode)
end

"""
Compile lines into a logistic circuit.
"""
function compile_logistic(lines::CircuitFormatLines, classes::Int)::LogisticCircuit
    # first compile a logic circuit
    logic_circuit, id2lognode = compile_smooth_logical_m(lines)
    decorate_logistic(lines, logic_circuit, classes, id2lognode)
end

"""
Compile circuit and vtree lines into a structured probabilistic circuit (one whose logic circuit origin is structured).
"""
function compile_struct_prob(circuit_lines::CircuitFormatLines, vtree_lines::VtreeFormatLines)
    logic_circuit, vtree, id2lognode, id2vtree = compile_smooth_struct_logical_m(circuit_lines, vtree_lines)
    prob_circuit = decorate_prob(circuit_lines, logic_circuit, id2lognode)
    return prob_circuit, vtree
end

function decorate_prob(lines::CircuitFormatLines, logic_circuit::LogicCircuit, id2lognode::Dict{ID,<:LogicCircuit})::ProbCircuit
    # set up cache mapping logic circuit nodes to their probabilistic decorator

    prob_circuit = ProbCircuit(logic_circuit)
    lognode2probnode = Dict{LogicCircuit, ProbCircuit}()

    prob_lin = linearize(prob_circuit) # TODO better implementation
    logic_lin = linearize(logic_circuit)

    foreach(i -> lognode2probnode[logic_lin[i]] = prob_lin[i], 1 : num_nodes(logic_circuit)) 

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
        @assert lnconstant(ln) == true
        root = id2probnode(ln.node_id)
        root.log_thetas .= [ln.weight, log1p(-exp(ln.weight))]
    end
    function compile(ln::DecisionLine{<:PSDDElement})
        root = id2probnode(ln.node_id)
        root.log_thetas .= [x.weight for x in ln.elements]
    end

    foreach(compile, lines)

    root
end

function decorate_logistic(lines::CircuitFormatLines, logic_circuit::LogicCircuit,
                            classes::Int, id2lognode::Dict{ID,<:LogicCircuit})::LogisticCircuit
                        
    # set up cache mapping logic circuit nodes to their logistic decorator
    logistic_circuit = LogisticCircuit(logic_circuit, classes)
    log2logistic = Dict{LogicCircuit, LogisticCircuit}()
    logistic_lin = linearize(logistic_circuit)
    logic_lin = linearize(logic_circuit)

    foreach(i -> log2logistic[logic_lin[i]] = logistic_lin[i], 1 : length(logic_lin)) 
    id2logisticnode(id) = log2logistic[id2lognode[id]]

    root = nothing
    # go through lines again and update the probabilistic circuit node parameters

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln into logistic circuit is not supported")
    end

    function compile(::Union{CircuitHeaderLine,CircuitCommentLine,UnweightedLiteralLine})
        # do nothing
    end

    function compile(ln::WeightedLiteralLine)
        root = id2logisticnode(ln.node_id)::Logistic⋁Node
        root.thetas[1, :] .= ln.weights
    end

    function compile(ln::DecisionLine{<:LCElement})
        root = id2logisticnode(ln.node_id)::Logistic⋁Node
        for (ind, elem) in enumerate(ln.elements)
            root.thetas[ind, :] .= elem.weights
        end
    end

    function compile(ln::BiasLine)
        root = id2logisticnode(ln.node_id)::Logistic⋁Node
        # @assert length(node.thetas) == 1
        root.thetas[1,:] .= ln.weights
    end

    foreach(compile, lines)

    root
end