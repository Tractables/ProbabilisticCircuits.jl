export learn_single_model
using LogicCircuits: split_step, struct_learn
using Statistics: mean
using Random
"""
Learn structure decomposable circuits
"""
function learn_single_model(train_x;
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=typemax(Int),
        seed=1337)

    # init
    Random.seed!(seed)
    pc, vtree = learn_struct_prob_circuit(train_x)

    # structure_update
    loss(circuit) = heuristic_loss(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var)
    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=loss, depth=depth, sanity_check=sanity_check)
        estimate_parameters(c, train_x; pseudocount=pseudocount)
        return c, missing
    end
    iter = 0
    log_per_iter(circuit) = begin
        ll = EVI(circuit, train_x)
        println("Log likelihood of iteration $iter is $(mean(ll))")
        println()
        iter += 1
        false
    end
    log_per_iter(pc)
    pc = struct_learn(pc; 
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()), 
        maxiter=maxiter, stop=log_per_iter)
end

