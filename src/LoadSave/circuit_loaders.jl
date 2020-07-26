export zoo_clt, zoo_clt_file, zoo_psdd, zoo_lc, load_prob_circuit, 
load_struct_prob_circuit, load_logistic_circuit

using LogicCircuits
using Pkg.Artifacts
using LogicCircuits.LoadSave: parse_psdd_file, parse_circuit_file, parse_vtree_file

#####################
# circuit loaders from module zoo
#####################

zoo_lc(name, num_classes) = 
    load_logistic_circuit(zoo_lc_file(name), num_classes)

zoo_clt_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.2/clts/$name"

zoo_clt(name) = 
    parse_clt(zoo_clt_file(name))

zoo_psdd(name) = 
    load_prob_circuit(zoo_psdd_file(name))

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
function load_prob_circuit(file::String)::ProbCircuit
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
function load_struct_prob_circuit(circuit_file::String, vtree_file::String)::Tuple{StructProbCircuit,PlainVtree}
    @assert endswith(circuit_file,".psdd")
    circuit_lines = parse_circuit_file(circuit_file)
    vtree_lines = parse_vtree_file(vtree_file)
    compile_struct_prob(circuit_lines, vtree_lines)
end

"""
Load a logistic circuit from file.
Support circuit file formats:
    * ".circuit" for logistic files
Supported vtree file formats:
    * ".vtree" for Vtree files
"""
function load_logistic_circuit(circuit_file::String, classes::Int)::LogisticCircuit
    @assert endswith(circuit_file,".circuit")
    circuit_lines = parse_circuit_file(circuit_file)
    compile_logistic(circuit_lines, classes)
end

#####################
# parse based on file extension
#####################

using MetaGraphs: MetaDiGraph, set_prop!, props, add_edge!

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

