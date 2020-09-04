
#####################
# Paramter Bit Circuits
#####################

"A `BitCircuit` with parameters attached to the elements"
struct ParamBitCircuit{V,M,W}
    bitcircuit::BitCircuit{V,M}
    params::W
end

import LogicCircuits: to_gpu, to_cpu #extend

to_gpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_gpu(c.bitcircuit), to_gpu(c.params))

to_cpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_cpu(c.bitcircuit), to_cpu(c.params))
