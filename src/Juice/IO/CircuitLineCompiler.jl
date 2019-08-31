#####################
# Compilers to Juice data structures starting from already parsed line objects
#####################

"""
Compile lines into a logical circuit
"""
compile_smooth_logical(lines::Vector{CircuitFormatLine})::UnstLogicalCircuit△ = 
    compile_smooth_logical_m(lines)[1]

"""
Compile lines into a logical circuit, while keeping track of id-to-node mappings
"""
function compile_smooth_logical_m(lines::Vector{CircuitFormatLine})

    # linearized circuit nodes
    circuit = Vector{UnstLogicalCircuitNode}()
    
    # mapping from node ids to node objects
    id2node = Dict{ID,UnstLogicalCircuitNode}()
    
    # literal cache is responsible for making leaf literal nodes unique and adding them to lin
    lit_cache = Dict{Lit,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!'
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    function compile(::TrimmedLine)
        error("Compiling a smooth circuit from a trimmed logical circuit file: this functionality is not yet implemented. 
               Instead parse as a non-smooth circuit and smooth afterwards.")
    end

    function compile(::Union{HeaderLine,CommentLine})
         # do nothing
    end
    function compile(ln::WeightedPosLiteralLine)
        id2node[ln.node_id] = literal_node(var2lit(ln.variable))
    end
    function compile(ln::WeightedNegLiteralLine)
        id2node[ln.node_id] = literal_node(-var2lit(ln.variable))
    end
    function compile(ln::UnweightedLiteralLine)
        id2node[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::WeightedTrueLeafLine)
        # because we promise to compile a smooth circuit, here we need to add a "smoothing or gate"
        n = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile_elements(e::NormalizedElement)
        n = ⋀Node([id2node[e.prime_id],id2node[e.sub_id]])
        push!(circuit,n)
        n
    end
    function compile(ln::DecisionLine{<:NormalizedElement})
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

    circuit, id2node
end

"""
Compile circuit and vtree lines into a structured logical circuit + vtree
"""
function compile_smooth_struct_logical(circuit_lines::Vector{CircuitFormatLine}, 
                                vtree_lines::Vector{VtreeFormatLine})
    compile_smooth_struct_logical_m(circuit_lines,vtree_lines)[1:2]
end

function compile_smooth_struct_logical_m(circuit_lines::Vector{CircuitFormatLine}, 
                                  vtree_lines::Vector{VtreeFormatLine})
    vtree, id2vtree = compile_vtree_format_lines_m(vtree_lines)
    circuit, id2circuit = compile_smooth_struct_logical_m(circuit_lines, id2vtree)
    circuit, vtree, id2vtree, id2circuit
end

"""
Compile lines into a structured logical circuit
"""
compile_smooth_struct_logical(lines::Vector{CircuitFormatLine}, 
                              id2vtree::Dict{ID, VtreeNode})::LogicalCircuit△ = 
    compile_smooth_struct_logical_m(lines, id2vtree)[1]

"""
Compile lines into a structured logical circuit, while keeping track of id-to-node mappings
"""
function compile_smooth_struct_logical_m(lines::Vector{CircuitFormatLine}, 
                                         id2vtree::Dict{ID, VtreeNode})

    # linearized circuit nodes
    circuit = Vector{StructLogicalCircuitNode}()
    
    # mapping from node ids to node objects
    id2node = Dict{ID,StructLogicalCircuitNode}()
    # literal cache is responsible for making leaf literal nodes unique and adding them to lin
    
    lit_cache = Dict{Lit,StructLogicalLeafNode}()
    literal_node(l::Lit, v::VtreeLeafNode) = get!(lit_cache, l) do
        leaf = (l>0 ? StructPosLeafNode(l,v) : StructNegLeafNode(-l,v)) #it's important for l to be a signed int!'
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    function compile(::TrimmedLine)
        error("Compiling a smooth circuit from a trimmed logical circuit file: this functionality is not yet implemented. 
               Instead parse as a non-smooth circuit and smooth afterwards.")
    end

    function compile(::Union{HeaderLine,CommentLine})
         # do nothing
    end
    function compile(ln::WeightedPosLiteralLine)
        id2node[ln.node_id] = literal_node(var2lit(ln.variable), id2vtree[ln.vtree_id])
    end
    function compile(ln::WeightedNegLiteralLine)
        id2node[ln.node_id] = literal_node(-var2lit(ln.variable), id2vtree[ln.vtree_id])
    end
    function compile(ln::NormalizedLiteralLine)
        id2node[ln.node_id] = literal_node(ln.literal, id2vtree[ln.vtree_id])
    end
    function compile(ln::WeightedTrueLeafLine)
        # because we promise to compile a smooth circuit, here we need to add an or gate
        vtree = id2vtree[ln.vtree_id]
        n = Struct⋁Node([literal_node(var2lit(ln.variable), vtree), literal_node(-var2lit(ln.variable), vtree)], vtree)
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile_elements(e::NormalizedElement, v::VtreeNode)
        n = Struct⋀Node([id2node[e.prime_id], id2node[e.sub_id]], v)
        push!(circuit,n)
        n
    end
    function compile(ln::DecisionLine{<:NormalizedElement})
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

    circuit, id2node
end

"""
Compile lines into a probabilistic circuit.
"""
function compile_prob(lines::Vector{CircuitFormatLine})::ProbCircuit△
    # first compile a logical circuit
    logical_circuit, id2lognode = compile_smooth_logical_m(lines)
    prob_circuit = decorate_prob(lines, logical_circuit, id2lognode)
    prob_circuit
end

"""
Compile lines into a structured probabilistic circuit (one whose logical circuit origin is structured).
"""
function compile_struct_prob(circuit_lines::Vector{CircuitFormatLine}, vtree_lines::Vector{VtreeFormatLine})::ProbCircuit△
    vtree, id2vtree = compile_vtree_format_lines_m(vtree_lines)
    logical_circuit, id2lognode = compile_smooth_struct_logical_m(circuit_lines, id2vtree)
    prob_circuit = decorate_prob(circuit_lines, logical_circuit, id2lognode)
    prob_circuit
end

function decorate_prob(lines::Vector{CircuitFormatLine}, logical_circuit::LogicalCircuit△, id2lognode::Dict{ID,<:LogicalCircuitNode})::ProbCircuit△
    # set up cache mapping logical circuit nodes to their probabilistic decorator
    lognode2probnode = ProbCache()
    # build a corresponding probabilistic circuit
    prob_circuit = ProbCircuit(logical_circuit,lognode2probnode)
    # map from line node ids to probabilistic circuit nodes
    id2probnode(id) = lognode2probnode[id2lognode[id]]

    # go through lines again and update the probabilistic circuit node parameters
    function compile(::Union{HeaderLine,CommentLine,NormalizedLiteralLine})
        # do nothing
    end
    function compile(ln::WeightedTrueLeafLine)
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= [ln.weight, log1p(-exp(ln.weight)) ]
    end
    function compile(ln::DecisionLine{<:PSDDElement})
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= [x.weight for x in ln.elements]
    end
    function compile(ln::BiasLine)
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= ln.weights
    end
    for ln in lines
        compile(ln)
    end

    prob_circuit
end