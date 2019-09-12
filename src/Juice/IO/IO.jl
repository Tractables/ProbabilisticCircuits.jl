module IO

using ..Logical
using ..Probabilistic

export

# CircuitParser
load_logical_circuit, load_smooth_logical_circuit, load_struct_smooth_logical_circuit, load_prob_circuit, load_struct_prob_circuit, load_psdd_logical_circuit,load_lc_logical_circuit, load_psdd_prob_circuit, parse_lc_file, parse_psdd_file,

# CircuitLineCompiler
# CircuitLineTypes
# can be kept internal

# CircuitSaver
save_as_dot,

# VtreeParser / Saver
parse_vtree_file, compile_vtree_format_lines, load_vtree, save

include("VtreeLineCompiler.jl")
include("VtreeParser.jl")
include("VtreeSaver.jl")

include("CircuitLineTypes.jl")
include("CircuitLineCompiler.jl")
include("CircuitParser.jl")

include("CircuitSaver.jl")

end