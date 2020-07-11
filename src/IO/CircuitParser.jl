
using MetaGraphs: MetaDiGraph, set_prop!, props

#####################
# general parser infrastructure for circuits
#####################

# The `ParserCombinator` library works correctly but is orders of magnitude too slow.
# Instead here we hardcode some simpler parsers to speed things up

"""
Load a probabilistic circuit from file.
Support circuit file formats:
 * ".psdd" for PSDD files
"""
function load_prob_circuit(file::String)::ProbΔ
    @assert endswith(file,".psdd")
    compile_prob(parse_psdd_file(file))
end

"""
Load a structured probabilistic circuit from file.
Support circuit file formats:
 * ".psdd" for PSDD files
Supported vtree file formats:
 * ".vtree" for Vtree files
"""
function load_struct_prob_circuit(circuit_file::String, vtree_file::String)::Tuple{ProbΔ,PlainVtree}
    @assert endswith(circuit_file,".psdd")
    circuit_lines = parse_circuit_file(circuit_file)
    vtree_lines = parse_vtree_file(vtree_file)
    compile_struct_prob(circuit_lines, vtree_lines)
end


function load_logistic_circuit(circuit_file::String, classes::Int)::LogisticΔ
    @assert endswith(circuit_file,".circuit")
    circuit_lines = parse_circuit_file(circuit_file)
    compile_logistic(circuit_lines, classes)
end


#####################
# parse based on file extension
#####################

function parse_circuit_file(file::String)::CircuitFormatLines
    if endswith(file,".circuit")
        parse_lc_file(file)
    elseif endswith(file,".psdd")
        parse_psdd_file(file)
    elseif endswith(file,".sdd")
        parse_sdd_file(file)
    else
        throw("Cannot parse this file type as a circuit: $file")
    end
end

"Parse a clt from given file"
function parse_clt(filename::String)::MetaDiGraph
    f = open(filename)
    n = parse(Int32,readline(f))
    n_root = parse(Int32,readline(f))
    clt = MetaDiGraph(n)
    for i in 1 : n_root
        root, prob = split(readline(f), " ")
        root, prob = parse(Int32, root), parse(Float64, prob)
        set_prop!(clt, root, :parent, 0)
        set_prop!(clt, root, :cpt, Dict(1=>prob,0=>1-prob))
    end

    for i = 1 : n - n_root
        dst, src, prob1, prob0 = split(readline(f), " ")
        dst, src, prob1, prob0 = parse(Int32, dst), parse(Int32, src), parse(Float64, prob1), parse(Float64, prob0)
        add_edge!(clt, src,dst)
        set_prop!(clt, dst, :parent, src)
        set_prop!(clt, dst, :cpt, Dict((1,1)=>prob1, (0,1)=>1-prob1, (1,0)=>prob0, (0,0)=>1-prob0))
    end
    return clt
end
