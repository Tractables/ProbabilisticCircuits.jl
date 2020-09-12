export sample

import Random: default_rng

"""
Sample from a PC without any evidence
"""
function sample(circuit::ProbCircuit; rng = default_rng())::BitVector
    
    inst = Dict{Var,Bool}()
    
    simulate(node) = simulate(node, GateType(node))
    
    simulate(node, ::LeafGate) = begin
        inst[variable(node)] = ispositive(node)
    end
    
    simulate(node, ::⋁Gate) = begin
        idx = sample_index(exp.(node.log_probs); rng)
        simulate(children(node)[idx])
    end

    simulate(node, ::⋀Gate) = 
        foreach(simulate, children(node))

    simulate(circuit)
    
    len = length(keys(inst))
    BitVector([inst[i] for i = 1:len])
end


"""
Uniformly sample based on the probability of the items and return the selected index
"""
function sample_index(probs::AbstractVector{<:Number}; rng = default_rng())::Int32
    z = sum(probs)
    q = rand(rng) * z
    cur = 0.0
    for i = 1:length(probs)
        cur += probs[i]
        if q <= cur
            return i
        end
    end
    return length(probs)
end
