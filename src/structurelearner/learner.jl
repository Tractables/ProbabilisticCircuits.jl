export learn_circuit,
       learn_circuit_miss

using LogicCircuits: split_step, struct_learn
using Statistics: mean
using Random

"""
Learn structure of a single structured decomposable circuit
"""
function learn_circuit(train_x; 
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        entropy_reg=0.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false,
        verbose=true,
        max_circuit_nodes=nothing)
    
    # Initial Structure
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    
    learn_circuit(train_x, pc, vtree; pick_edge, pick_var, 
                           depth, pseudocount, sanity_check,
                           maxiter, seed, return_vtree, entropy_reg, verbose, max_circuit_nodes)
end


"""
Learn structure of a single structured decomposable circuit from missing data. 

Missing feature are denoted by `missing`. 
Median Imputation is used by default for initial structure (set `impute_method` for other options).
"""
function learn_circuit_miss(train_x; 
        impute_method::Symbol=:median,
        pick_edge="eFlow", depth=1, 
        pseudocount=1.0,
        entropy_reg=0.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false,
        verbose=true,
        max_circuit_nodes=nothing)
    
    # Initial Structure
    train_x_impute = impute(train_x; method=impute_method)
    pc, vtree = learn_chow_liu_tree_circuit(train_x_impute)

    # Only vRand supported for missing data
    pick_var="vRand"
    
    learn_circuit(train_x, pc, vtree; pick_edge, pick_var, 
                  depth, pseudocount, sanity_check, 
                  maxiter, seed, return_vtree, entropy_reg,
                  max_circuit_nodes, verbose,
                  has_missing=true)
end



function learn_circuit(train_x, pc, vtree;
        pick_edge="eFlow", pick_var="vMI", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false,
        batch_size=0,
        splitting_data=nothing,
        use_gpu=false,
        entropy_reg=0.0,
        verbose=true,
        max_circuit_nodes=nothing,
        has_missing::Bool=false
        )

    if seed !== nothing
        Random.seed!(seed)
    end

    if has_missing
        estimate_parameters_func = estimate_parameters_em
        likelihood_avg_func = marginal_log_likelihood_avg
        LogLikelihood_str = "Marginal LogLikelihood"
    else
        estimate_parameters_func = estimate_parameters
        likelihood_avg_func = log_likelihood_avg
        LogLikelihood_str = "LogLikelihood"
    end



    # structure_update
    loss(circuit) = heuristic_loss(circuit, 
                                    splitting_data == nothing ? train_x : splitting_data; 
                                    pick_edge=pick_edge, 
                                    pick_var=pick_var,
                                    miss_data=has_missing)
    

    pc_split_step(circuit) = begin
        r = split_step(circuit; loss=loss, depth=depth, sanity_check=sanity_check)
        if isnothing(r) return nothing end
        c, = r
        if batch_size > 0
            estimate_parameters_func(c, batch(train_x, batch_size); pseudocount, use_gpu, entropy_reg)
        else
            estimate_parameters_func(c, train_x; pseudocount, use_gpu, entropy_reg)
        end
        return c, missing
    end
    iter = 0
    log_per_iter(circuit) = begin
        if batch_size > 0
            ll = likelihood_avg_func(circuit, batch(train_x, batch_size); use_gpu)
        else
            ll = likelihood_avg_func(circuit, train_x; use_gpu)
        end
        verbose && println("Iteration $iter/$maxiter. $(LogLikelihood_str) = $(ll); nodes = $(num_nodes(circuit)); edges =  $(num_edges(circuit)); params = $(num_parameters(circuit))")
        iter += 1
        
        if !isnothing(max_circuit_nodes) && num_nodes(circuit) > max_circuit_nodes
            epoch_printer("Stopping early, circuit node count ($(num_nodes(circuit))) is above max threshold $(max_circuit_nodes).");
            return true; # stop
        end

        false
    end

    # Log Before Learning
    log_per_iter(pc)

    pc = struct_learn(pc; 
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()), 
        maxiter=maxiter, stop=log_per_iter, verbose=verbose)

    if return_vtree
        pc, vtree
    else
        pc
    end
end