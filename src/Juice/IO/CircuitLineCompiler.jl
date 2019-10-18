#####################
# Compilers to Juice data structures starting from already parsed line objects
#####################

"""
Compile lines into a unstructured logical circuit
"""
compile_logical(lines::CircuitFormatLines)::UnstLogicalCircuit = 
    compile_logical_m(lines)[1]

"""
Compile lines into a unstructured logical circuit, 
while keeping track of id-to-node mappings
"""
function compile_logical_m(lines::CircuitFormatLines)

    # linearized circuit nodes
    circuit = Vector{UnstLogicalΔNode}()
    
    # mapping from circuit node ids to node objects
    id2node = Dict{ID,UnstLogicalΔNode}()
    
    # literal cache is responsible for making leaf literal nodes unique and adding them to `circuit`
    lit_cache = Dict{Lit,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = LiteralNode(l)
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    seen_true = false
    seen_false = false

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine})
         # do nothing
    end
    function compile(ln::LiteralLine)
        id2node[ln.node_id] = literal_node(literal(ln))
    end
    function compile(ln::ConstantLine)
        if constant(ln) == true
            n = TrueNode()
            seen_true || (push!(circuit,n); seen_true = true)
        else
            n = FalseNode()
            seen_false || (push!(circuit,n); seen_false = true)
        end
        id2node[ln.node_id] = n
    end
    function compile_elements(e::Element)
        n = ⋀Node([id2node[e.prime_id],id2node[e.sub_id]])
        push!(circuit,n)
        n
    end
    function compile(ln::DecisionLine)
        n = ⋁Node(map(compile_elements, ln.elements))
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile(ln::BiasLine)
        n = ⋁Node([circuit[end]])
        push!(circuit,n)
        id2node[ln.node_id] = n
    end

    for ln in lines
        compile(ln)
    end

    return circuit, id2node
end

#TODO add compile_struct_logical

"""
Compile lines into a smooth unstructured logical circuit
"""
compile_smooth_logical(lines::CircuitFormatLines)::UnstLogicalCircuit = 
    compile_smooth_logical_m(lines)[1]

"""
Compile lines into a smooth unstructured logical circuit, 
while keeping track of id-to-node mappings
"""
function compile_smooth_logical_m(lines::CircuitFormatLines)

    # linearized circuit nodes
    circuit = Vector{UnstLogicalΔNode}()
    
    # mapping from circuit node ids to node objects
    id2node = Dict{ID,UnstLogicalΔNode}()
    
    # literal cache is responsible for making leaf literal nodes unique and adding them to `circuit`
    lit_cache = Dict{Lit,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = LiteralNode(l)
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    smoothing_warning = "Cannot compile a smooth logical circuit from lines that are not normalized: there is no way to smooth without knowing the variable scope. Instead compile a non-smooth logical circuit and smooth it afterwards."

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine})
         # do nothing
    end
    function compile(ln::LiteralLine)
        @assert is_normalized(ln) " $smoothing_warning"
        id2node[ln.node_id] = literal_node(literal(ln))
    end

    function compile(ln::WeightedNamedConstantLine)
        @assert constant(ln) == true
        # because we promise to compile a smooth circuit, here we need to add a "smoothing or gate"
        n = ⋁Node([literal_node(var2lit(variable(ln))), 
                    literal_node(-var2lit(variable(ln)))])
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile(ln::AnonymousConstantLine)
        error(smoothing_warning)
    end
    function compile_elements(e::TrimmedElement)
        error(smoothing_warning)
    end
    function compile_elements(e::NormalizedElement)
        n = ⋀Node([id2node[e.prime_id],id2node[e.sub_id]])
        push!(circuit,n)
        n
    end
    function compile(ln::DecisionLine)
        n = ⋁Node(map(compile_elements, ln.elements))
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile(ln::BiasLine)
        n = ⋁Node([circuit[end]])
        push!(circuit,n)
        id2node[ln.node_id] = n
    end

    for ln in lines
        compile(ln)
    end

    return circuit, id2node
end

"""
Compile circuit and vtree lines into a structured logical circuit with its vtree
"""
function compile_smooth_struct_logical(circuit_lines::CircuitFormatLines, 
                                vtree_lines::VtreeFormatLines)
    compile_smooth_struct_logical_m(circuit_lines,vtree_lines)[1:2]
end

"""
Compile circuit and vtree lines into a structured logical circuit with its vtree, 
while keeping track of id-to-node mappings
"""
function compile_smooth_struct_logical_m(circuit_lines::CircuitFormatLines, 
                                  vtree_lines::VtreeFormatLines)
    vtree, id2vtree = compile_vtree_format_lines_m(vtree_lines)
    circuit, id2circuit = compile_smooth_struct_logical_m(circuit_lines, id2vtree)
    return circuit, vtree, id2vtree, id2circuit
end

"""
Compile circuit lines and vtree node mapping into a structured logical circuit, 
while keeping track of id-to-node mappings
"""
function compile_smooth_struct_logical_m(lines::CircuitFormatLines, 
                                         id2vtree::Dict{ID, VtreeNode})

    # linearized circuit nodes
    circuit = Vector{StructLogicalΔNode}()
    
    # mapping from node ids to node objects
    id2node = Dict{ID,StructLogicalΔNode}()

    # literal cache is responsible for making leaf literal nodes unique and adding them to `circuit`
    lit_cache = Dict{Lit,StructLogicalLeafNode}()
    literal_node(l::Lit, v::VtreeLeafNode) = get!(lit_cache, l) do
        leaf = StructLiteralNode(l,v)
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    smoothing_warning = "Cannot compile a smooth logical circuit from lines that are not normalized: functionality to determine variable scope and perform smoothing not implemented in the line compiler.  Instead compile a non-smooth logical circuit and smooth it afterwards."

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine})
         # do nothing
    end
    function compile(ln::LiteralLine)
        @assert is_normalized(ln) smoothing_warning
        id2node[ln.node_id] = literal_node(ln.literal, id2vtree[ln.vtree_id])
    end

    function compile(ln::ConstantLine)
        vtree = id2vtree[ln.vtree_id]
        if is_normalized(ln)
            variable = (vtree::VtreeLeafNode).var
            @assert !(ln isa WeightedNamedConstantLine) || variable == ln.variable "Vtree mapping must agree with variable field of circuit line"
        else
            error(smoothing_warning)
        end
        if constant(ln) == true
            # because we promise to compile a smooth circuit, here we need to add an or gate
            n = Struct⋁Node([literal_node(var2lit(variable), vtree), literal_node(-var2lit(variable), vtree)], vtree)
        else
            error("False leaf logical circuit nodes not yet implemented")
        end
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile_elements(e::TrimmedElement, ::VtreeNode)
        error(smoothing_warning)
    end
    function compile_elements(e::NormalizedElement, v::VtreeNode)
        n = Struct⋀Node([id2node[e.prime_id], id2node[e.sub_id]], v)
        push!(circuit,n)
        n
    end
    function compile(ln::DecisionLine)
        vtree = id2vtree[ln.vtree_id]
        n = Struct⋁Node(map(e -> compile_elements(e, vtree), ln.elements), vtree)
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile(ln::BiasLine)
        n = Struct⋁Node([circuit[end]], circuit[end].vtree)
        push!(circuit,n)
        id2node[ln.node_id] = n
    end

    for ln in lines
        compile(ln)
    end

    return circuit, id2node
end

"""
Compile lines into a probabilistic circuit.
"""
function compile_prob(lines::CircuitFormatLines)::ProbCircuit
    # first compile a logical circuit
    logical_circuit, id2lognode = compile_smooth_logical_m(lines)
    decorate_prob(lines, logical_circuit, id2lognode)
end

"""
Compile lines into a logistic circuit.
"""
function compile_logistic(lines::CircuitFormatLines, classes::Int)::LogisticCircuit
    # first compile a logical circuit
    logical_circuit, id2lognode = compile_smooth_logical_m(lines)
    decorate_logistic(lines, logical_circuit, classes, id2lognode)
end

"""
Compile circuit and vtree lines into a structured probabilistic circuit (one whose logical circuit origin is structured).
"""
function compile_struct_prob(circuit_lines::CircuitFormatLines, vtree_lines::VtreeFormatLines)
    logical_circuit, vtree, id2vtree, id2lognode = compile_smooth_struct_logical_m(circuit_lines, vtree_lines)
    prob_circuit = decorate_prob(circuit_lines, logical_circuit, id2lognode)
    return prob_circuit, vtree
end

function decorate_prob(lines::CircuitFormatLines, logical_circuit::LogicalCircuit, id2lognode::Dict{ID,<:LogicalΔNode})::ProbCircuit
    # set up cache mapping logical circuit nodes to their probabilistic decorator
    lognode2probnode = ProbCache()
    # build a corresponding probabilistic circuit
    prob_circuit = ProbCircuit(logical_circuit,lognode2probnode)
    # map from line node ids to probabilistic circuit nodes
    id2probnode(id) = lognode2probnode[id2lognode[id]]

    # go through lines again and update the probabilistic circuit node parameters

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln into probabilistic circuit is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine,UnweightedLiteralLine})
        # do nothing
    end
    function compile(ln::WeightedNamedConstantLine)
        @assert constant(ln) == true
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= [ln.weight, log1p(-exp(ln.weight)) ]
    end
    function compile(ln::DecisionLine{<:PSDDElement})
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= [x.weight for x in ln.elements]
    end
    for ln in lines
        compile(ln)
    end

    prob_circuit
end


function decorate_logistic(lines::CircuitFormatLines, logical_circuit::LogicalCircuit, 
                            classes::Int, id2lognode::Dict{ID,<:LogicalΔNode})::LogisticCircuit
                        
    # set up cache mapping logical circuit nodes to their logistic decorator
    log2logistic = LogisticCache()
    # build a corresponding probabilistic circuit
    logistic_circuit = LogisticCircuit(logical_circuit, classes, log2logistic)
    # map from line node ids to probabilistic circuit nodes
    id2logisticnode(id) = log2logistic[id2lognode[id]]

    # go through lines again and update the probabilistic circuit node parameters

    function compile(ln::CircuitFormatLine)
        error("Compilation of line $ln into logistic circuit is not supported")
    end
    function compile(::Union{CircuitHeaderLine,CircuitCommentLine,UnweightedLiteralLine})
        # do nothing
    end

    function compile(ln::CircuitHeaderLine)
        # do nothing
    end

    function compile(ln::WeightedLiteralLine)
        node = id2logisticnode(ln.node_id)::LogisticLiteral
        node.thetas .= ln.weights
    end

    function compile(ln::DecisionLine{<:LCElement})
        node = id2logisticnode(ln.node_id)::Logistic⋁
        for (ind, elem) in enumerate(ln.elements)
            node.thetas[ind, :] .= elem.weights
        end
    end

    function compile(ln::BiasLine)
        node = id2logisticnode(ln.node_id)::Logistic⋁
        # @assert length(node.thetas) == 1
        node.thetas[1,:] .= ln.weights
    end

    for ln in lines
        compile(ln)
    end

    logistic_circuit
end