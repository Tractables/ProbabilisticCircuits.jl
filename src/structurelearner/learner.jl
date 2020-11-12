export learn_circuit
using LogicCircuits: split_step, struct_learn
using Statistics: mean
using Random
"""
Learn structure of a single structured decomposable circuit
"""
function learn_circuit(train_x; 
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false)
    
    # Initial Structure
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    
    learn_circuit(train_x, pc, vtree; pick_edge, pick_var, depth, pseudocount, sanity_check, 
                  maxiter, seed, return_vtree)
end
function learn_circuit(train_x, pc, vtree;
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false)

    if seed !== nothing
        Random.seed!(seed)
    end

    # structure_update
    loss(circuit) = heuristic_loss(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var)
    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=loss, depth=depth, sanity_check=sanity_check)
        estimate_parameters(c, train_x; pseudocount=pseudocount)
        return c, missing
    end
    iter = 0
    log_per_iter(circuit) = begin
        ll = EVI(circuit, train_x);
        println("Iteration $iter/$maxiter. LogLikelihood = $(mean(ll)); nodes = $(num_nodes(circuit)); edges =  $(num_edges(circuit)); params = $(num_parameters(circuit))")
        iter += 1
        false
    end
    log_per_iter(pc)
    pc = struct_learn(pc; 
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()), 
        maxiter=maxiter, stop=log_per_iter)

    if return_vtree
        pc, vtree
    else
        pc
    end
end

