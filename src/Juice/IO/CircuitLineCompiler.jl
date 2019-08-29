#####################
# Compilers to Juice data structures starting from already parsed line objects
#####################

abstract type CircuitFormatLine <: FormatLine end

struct CommentLine{T<:AbstractString} <: CircuitFormatLine
    comment::T
end

struct HeaderLine <: CircuitFormatLine end

abstract type AbstractLiteralLine <: CircuitFormatLine end

struct WeightedPosLiteralLine <: AbstractLiteralLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct WeightedNegLiteralLine <: AbstractLiteralLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct LiteralLine <: AbstractLiteralLine
    node_id::UInt32
    vtree_id::UInt32
    literal::Lit
end

struct TrueLeafLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weight::Float32
end

abstract type ElementTuple end

struct LCElementTuple <: ElementTuple
    prime_id::UInt32
    sub_id::UInt32
    weights::Vector{Float32}
end

struct PSDDElementTuple <: ElementTuple
    prime_id::UInt32
    sub_id::UInt32
    weight::Float32
end

abstract type DecisionLine <: CircuitFormatLine end

struct LCDecisionLine <: DecisionLine
    node_id::UInt32
    vtree_id::UInt32
    num_elements::UInt32
    elements:: Vector{ElementTuple}
end

struct PSDDDecisionLine <: DecisionLine
    node_id::UInt32
    vtree_id::UInt32
    num_elements::UInt32
    elements:: Vector{PSDDElementTuple}
end

struct BiasLine <: CircuitFormatLine
    node_id::UInt32
    weights::Vector{Float32}
    BiasLine(weights) = new(typemax(UInt32), weights)
end


"""
Compile lines into a logical circuit
"""
compile_lines_logical(lines::Vector{CircuitFormatLine})::LogicalCircuit△ = 
    compile_lines_logical_with_mapping(lines)[1]

"""
Compile lines into a logical circuit, while keeping track of id-to-node mappings
"""
function compile_lines_logical_with_mapping(lines::Vector{CircuitFormatLine})

    # linearized circuit nodes
    circuit = Vector{LogicalCircuitNode}()
    # mapping from node ids to node objects
    id2node = Dict{UInt32,CircuitNode}()
    # literal cache is responsible for making leaf literal nodes unique and adding them to lin
    lit_cache = Dict{Int32,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!'
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
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
    function compile(ln::LiteralLine)
        id2node[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::TrueLeafLine)
        n = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        push!(circuit,n)
        id2node[ln.node_id] = n
    end
    function compile_elements(e::ElementTuple)
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

    circuit, id2node
end

"""
Compile lines into a probabilistic circuit
"""
function compile_lines_prob(lines::Vector{CircuitFormatLine})::ProbCircuit△
    # first compile a logical circuit
    logical_circuit, id2lognode = compile_lines_logical_with_mapping(lines)
    # set up cache mapping logical circuit nodes to their probabilistic decorator
    lognode2probnode = ProbCache()
    # build a corresponding probabilistic circuit
    prob_circuit = ProbCircuit(logical_circuit,lognode2probnode)
    # map from line node ids to probabilistic circuit nodes
    id2probnode(id) = lognode2probnode[id2lognode[id]]

    # go through lines again and update the probabilistic circuit node parameters
    compile(::Union{HeaderLine,CommentLine,AbstractLiteralLine}) = () # do nothing
    function compile(ln::TrueLeafLine)
        node = id2probnode(ln.node_id)::Prob⋁
        node.log_thetas .= [ln.weight, log1p(-exp(ln.weight)) ]
    end
    function compile(ln::DecisionLine)
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