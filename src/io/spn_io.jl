struct SpnFormat <: FileFormat end

##############################################
# Read SPN format (some dialect of the Libra AC format?)
##############################################

const spn_grammar = raw"""
    start : domains (_NL node)+ _NL "EOF" _NL?

    domains : "(2" (_WS "2")* ")"
    
    node : "v" _WS INT _WS INT -> literal_node
         | "*" (_WS INT)+ -> prod_node
         | "+" (_WS INT _WS LOGPROB)+ -> sum_node
         
    %import common.INT
    %import common.SIGNED_NUMBER -> LOGPROB
    %import common.WS_INLINE -> _WS
    %import common.NEWLINE -> _NL
    """

const spn_parser() = Lark(spn_grammar)

struct SpnParse <: PCTransformer
    nodes::Vector{PlainProbCircuit}
    SpnParse() = new(PlainProbCircuit[])
end

@rule start(t::SpnParse, x) =
    t.nodes[end]

@inline domains(t::SpnParse, x) = begin 
    d = Base.parse.(Int,x)
    @assert d .== 2 "Only Boolean domains are currently supported, not $d in SPN format"
    d
end

@rule literal_node(t::SpnParse, x) = begin
    var = Base.parse(Var,x[1]) + 1
    @assert x[2] == "0" || x[2] == "1" "Boolean domains only." 
    sign = (x[2] == "1")
    push!(t.nodes, PlainInputNode(var, LiteralDist(sign)))
end

@rule prod_node(t::SpnParse,x) = begin
    child_i = Base.parse.(Int,x) .+ 1
    children = t.nodes[child_i]
    push!(t.nodes, PlainMulNode(children))
end

@rule sum_node(t::SpnParse,x) = begin
    child_i = Base.parse.(Int,x[1:2:end]) .+ 1
    children = t.nodes[child_i]
    log_probs = Base.parse.(Float64,x[2:2:end])
    push!(t.nodes, PlainSumNode(children, log_probs))
end

function Base.parse(::Type{PlainProbCircuit}, str, ::SpnFormat) 
    ast = Lerche.parse(spn_parser(), str)
    Lerche.transform(SpnParse(), ast)
end

Base.read(io::IO, ::Type{PlainProbCircuit}, ::SpnFormat) =
    parse(PlainProbCircuit, read(io, String), SpnFormat())


##############################################
# Write SPNs
##############################################

function Base.write(io::IO, circuit::ProbCircuit, ::SpnFormat)

    labeling = label_nodes(circuit)
    map!(x -> x-1, values(labeling)) # nodes are 0-based indexed

    println(io, "(2" * " 2"^(num_randvars(circuit)-1) * ")")
    foreach(circuit) do n
        if isinput(n)
            state = value(dist(n)) ? "1" : "0"
            println(io, "v $(randvar(n)-1) $state")
        else
            print(io, ismul(n) ? "*" : "+")
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
    println(io, "EOF")
    nothing
end