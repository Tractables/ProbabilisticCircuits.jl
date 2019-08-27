#####################
# Compilers to Juice data structures starting from already parsed line types
#####################

abstract type CircuitFormatLine end

struct CommentLine{T<:AbstractString} <: CircuitFormatLine
    comment::T
end

struct HeaderLine <: CircuitFormatLine end

struct PosLiteralLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct NegLiteralLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct LiteralLine <: CircuitFormatLine
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


function compile_circuit_format_lines(lines::Vector{CircuitFormatLine})::Vector{LogicalCircuitNode}
    lin = Vector{CircuitNode}()
    node_cache = Dict{UInt32,CircuitNode}()

    #  literal cache is responsible for making leaf nodes unique and adding them to lin
    lit_cache = Dict{Int32,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!'
        push!(lin,leaf)
        leaf
    end

    compile(::Union{HeaderLine,CommentLine}) = () # do nothing
    function compile(ln::PosLiteralLine)
        node_cache[ln.node_id] = literal_node(var2lit(ln.variable))
    end
    function compile(ln::NegLiteralLine)
        node_cache[ln.node_id] = literal_node(-var2lit(ln.variable))
    end
    function compile(ln::LiteralLine)
        node_cache[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::TrueLeafLine)
        n = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        push!(lin,n)
        node_cache[ln.node_id] = n
    end
    function compile_elements(e::ElementTuple)
        n = ⋀Node([node_cache[e.prime_id],node_cache[e.sub_id]])
        push!(lin,n)
        n
    end
    function compile(ln::DecisionLine)
        n = ⋁Node(map(compile_elements, ln.elements))
        push!(lin,n)
        node_cache[ln.node_id] = n
    end
    function compile(ln::BiasLine)
        n = ⋁Node([lin[end]])
        push!(lin,n)
        node_cache[ln.node_id] = n
    end

    for ln in lines
        compile(ln)
    end

    lin
end

function compile_prob_circuit_format_lines(lines::Vector{CircuitFormatLine})::Vector{ProbCircuitNode}
    lin = Vector{ProbCircuitNode}()
    node_cache = Dict{UInt32, CircuitNode}()
    prob_cache = ProbCache()

    #  literal cache is responsible for making leaf nodes unique and adding them to lin
    lit_cache = Dict{Lit, LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!
        prob_leaf = (l > 0 ? ProbPosLeaf(leaf) : ProbNegLeaf(leaf))
        push!(lin, prob_leaf)
        leaf
    end

    compile(::Union{HeaderLine,CommentLine}) = () # do nothing
    function compile(ln::PosLiteralLine)
        node_cache[ln.node_id] = literal_node(var2lit(ln.variable))
    end
    function compile(ln::NegLiteralLine)
        node_cache[ln.node_id] = literal_node(-var2lit(ln.variable))
    end
    function compile(ln::LiteralLine)
        node_cache[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::TrueLeafLine)
        temp = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        n.log_thetas .= 0
        n.log_thetas .+= [ln.weight, log(1-exp(ln.weight) + 1e-300) ]
        node_cache[ln.node_id] = temp
    end
    function compile_elements(e::ElementTuple)
        temp = ⋀Node([node_cache[e.prime_id],node_cache[e.sub_id]])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        temp
    end
    function compile(ln::DecisionLine)
        temp = ⋁Node(map(compile_elements, ln.elements))
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        n.log_thetas .= 0
        n.log_thetas .+= [x.weight for x in ln.elements]
        push!(lin,n)
        node_cache[ln.node_id] = temp
    end
    function compile(ln::BiasLine)
        temp = ⋁Node([lin[end]])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        n.log_thetas .= 0
        n.log_thetas .+= ln.weights
        node_cache[ln.node_id] = temp
    end

    for ln in lines
        compile(ln)
    end

    # Sanity Check
    for node in lin
        if node isa Prob⋁
            if sum(isnan.(node.log_thetas)) > 0
                throw("There is a NaN in one of the log_thetas")
            end
        end
    end

    lin
end