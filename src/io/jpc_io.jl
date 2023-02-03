struct JpcFormat <: FileFormat end

##############################################
# Read JPC (Juice Probabilistic Circuit)
##############################################

const jpc_grammar = raw"""
    start: header (_NL node)+ _NL?

    header : "jpc" _WS INT
    
    node : "L" _WS INT _WS INT _WS SIGNED_INT -> literal_node
         | "I" _WS INT _WS INT _WS INT _WS INT -> indicator_node
         | "B" _WS INT _WS INT _WS INT _WS INT _WS LOGPROB -> binomial_node
         | "G" _WS INT _WS INT _WS INT _WS LOGPROB _WS LOGPROB -> gaussian_node
         | "C" _WS INT _WS INT _WS INT (_WS LOGPROB)+ -> categorical_node
         | "P" _WS INT _WS INT _WS INT child_nodes -> prod_node
         | "S" _WS INT _WS INT _WS INT weighted_child_nodes -> sum_node
         
    child_nodes : (_WS INT)+
    weighted_child_nodes: (_WS INT _WS LOGPROB)+
    
    %import common.INT
    %import common.SIGNED_INT
    %import common.SIGNED_NUMBER -> LOGPROB
    %import common.WS_INLINE -> _WS
    %import common.NEWLINE -> _NL
    """ * dimacs_comments

jpc_parser() = Lark(jpc_grammar)

abstract type JpcParse <: PCTransformer end

@inline_rule header(t::JpcParse, x) = 
    Base.parse(Int,x)

@rule start(t::JpcParse, x) = begin
    @assert num_nodes(x[end]) == x[1]
    x[end]
end 

@rule child_nodes(t::JpcParse, x) = 
    map(id -> t.nodes[id], x)

@rule weighted_child_nodes(t::JpcParse, x) = begin
    children = map(id -> t.nodes[id], x[1:2:end])
    log_probs = Base.parse.(Float64,x[2:2:end])
    (children, log_probs)
end

#  parse unstructured
struct PlainJpcParse <: JpcParse
    nodes::Dict{String,PlainProbCircuit}
    PlainJpcParse() = new(Dict{String,PlainProbCircuit}())
end

@rule literal_node(t::PlainJpcParse, x) = begin
    lit = Base.parse(Int,x[3])
    var = abs(lit)
    sign = lit > 0
    t.nodes[x[1]] = PlainInputNode(var, Literal(sign))
end

@rule indicator_node(t::PlainJpcParse, x) = begin
    var = Base.parse(Int,x[3])
    value = Base.parse(Int,x[4])
    t.nodes[x[1]] = PlainInputNode(var, Indicator(value))
end

@rule gaussian_node(t::PlainJpcParse, x) = begin
    var = Base.parse(Int,x[3])
    mu = Base.parse(Float64, x[4])
    sigma = Base.parse(Float64, x[5])
    t.nodes[x[1]] = PlainInputNode(var, Gaussian(mu, sigma))
end

@rule binomial_node(t::PlainJpcParse, x) = begin
    var = Base.parse(Int,x[3])
    N = Base.parse(UInt32, x[4])
    logp = Base.parse(Float64, x[5])
    t.nodes[x[1]] = PlainInputNode(var, Binomial(N, exp(logp)))
end

@rule categorical_node(t::PlainJpcParse, x) = begin
    var = Base.parse(Int,x[3])
    log_probs = Base.parse.(Float64, x[4:end])
    t.nodes[x[1]] = PlainInputNode(var, Categorical(log_probs))
end

@rule prod_node(t::PlainJpcParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    t.nodes[x[1]] = PlainMulNode(x[4])
end

@rule sum_node(t::PlainJpcParse,x) = begin
    @assert length(x[4][1]) == length(x[4][2]) == Base.parse(Int,x[3])
    t.nodes[x[1]] = PlainSumNode(x[4][1], x[4][2])
end

function Base.parse(::Type{PlainProbCircuit}, str, ::JpcFormat) 
    ast = Lerche.parse(jpc_parser(), str)
    Lerche.transform(PlainJpcParse(), ast)
end

function Base.read(io::IO, ::Type{PlainProbCircuit}, ::JpcFormat, fast = true)
    if fast
        read_fast(io, PlainProbCircuit, JpcFormat())
    else
        parse(PlainProbCircuit, read(io, String), JpcFormat())
    end
end
    
# fast brittle read
function read_fast(input, ::Type{<:ProbCircuit} = PlainProbCircuit, ::JpcFormat = JpcFormat())
    # would be better using `Parsers.jl` but that package lacks documentation`
    nodes = PlainProbCircuit[]
    for line in eachline(input)
        if startswith(line, "c")
            # do nothing
        else
            tokens = split(line, " ")
            if startswith(line, "jpc")
                num_nodes = Base.parse(Int,tokens[2])
                nodes = Vector{PlainProbCircuit}(undef,num_nodes)
            else
                id = Base.parse(Int,tokens[2]) + 1
                if startswith(line, "L")
                    lit = Base.parse(Int,tokens[4])
                    var = abs(lit)
                    sign = lit > 0
                    nodes[id] = PlainInputNode(var, Literal(sign))
                elseif startswith(line, "I")
                    var = Base.parse(Int,tokens[4])
                    val = Base.parse(Int,tokens[5])
                    nodes[id] = PlainInputNode(var, Indicator(val))
                elseif startswith(line, "C")
                    var = Base.parse(Int,tokens[4])
                    log_probs = Base.parse.(Float64, tokens[5:end])
                    nodes[id] = PlainInputNode(var, Categorical(log_probs))
                elseif startswith(line, "B")
                    var = Base.parse(Int,tokens[4])
                    N = Base.parse(UInt32, tokens[5])
                    logp = Base.parse(Float64, tokens[6])
                    nodes[id] = PlainInputNode(var, Binomial(N, exp(logp)))
                elseif startswith(line, "G")
                    var = Base.parse(Int,tokens[4])
                    mu = Base.parse(Float64, tokens[5])
                    sigma = Base.parse(Float64, tokens[6])
                    nodes[id] = PlainInputNode(var, Gaussian(mu, sigma))
                elseif startswith(line, "P")
                    child_ids = Base.parse.(Int, tokens[5:end]) .+ 1
                    children = nodes[child_ids]
                    nodes[id] = PlainMulNode(children)
                elseif startswith(line, "S")
                    child_ids = Base.parse.(Int, tokens[5:2:end]) .+ 1
                    children = nodes[child_ids]
                    log_probs = Base.parse.(Float64, tokens[6:2:end])
                    nodes[id] = PlainSumNode(children, log_probs)
                else
                    error("Cannot parse line: $line")
                end
            end
        end
    end
    nodes[end]
end

##############################################
# Write JPCs
##############################################

const JPC_FORMAT = """c this file was saved by ProbabilisticCircuits.jl
c ids of jpc nodes start at 0
c jpc nodes appear bottom-up, children before parents
c
c file syntax:
c jpc count-of-jpc-nodes
c L id-of-jpc-node id-of-vtree literal
c I id-of-jpc-node id-of-vtree variable indicator-value
c C id-of-jpc-node id-of-vtree variable {log-probability}+
c B id-of-jpc-node id-of-vtree variable binomial-N binomial-P
c G id-of-jpc-node id-of-vtree variable gaussian-mu gaussian-sigma
c P id-of-sum-jpc-node id-of-vtree number-of-children {child-id}+
c S id-of-product-jpc-node id-of-vtree number-of-children {child-id log-probability}+
c"""

function Base.write(io::IO, circuit::ProbCircuit, ::JpcFormat, vtreeid::Function = (x -> 0))

    labeling = label_nodes(circuit)
    map!(x -> x-1, values(labeling)) # vtree nodes are 0-based indexed

    println(io, JPC_FORMAT)
    println(io, "jpc $(num_nodes(circuit))")
    foreach(circuit) do n
        if isinput(n)
            var = randvar(n)
            d = dist(n)
            if d isa Literal
                literal = value(d) ? var : -var
                println(io, "L $(labeling[n]) $(vtreeid(n)) $literal")
            elseif d isa Indicator{<:Integer}
                println(io, "I $(labeling[n]) $(vtreeid(n)) $var $(value(d))")
            elseif d isa Categorical
                print(io, "C $(labeling[n]) $(vtreeid(n)) $var")
                foreach(p -> print(io, " $p"), params(d))
                println(io)
            elseif d isa Binomial
                print(io, "B $(labeling[n]) $(vtreeid(n)) $var $(d.N) $(log(d.p))")
                println(io)
            elseif d isa Gaussian
                print(io, "G $(labeling[n]) $(vtreeid(n)) $var $(d.mu) $(d.sigma)")
                println(io)
            else
                error("Input distribution type $(typeof(d)) is unknown to the JPC file format")
            end
        else
            t = ismul(n) ? "P" : "S"
            print(io, "$t $(labeling[n]) $(vtreeid(n)) $(num_inputs(n))")
            if ismul(n)
                for child in inputs(n)
                    print(io, " $(labeling[child])")
                end
            else
                @assert issum(n)  
                for (child, logp) in zip(inputs(n), params(n))
                    print(io, " $(labeling[child]) $logp")
                end    
            end
            println(io)
        end
    end
    nothing
end