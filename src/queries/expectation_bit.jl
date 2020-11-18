using CUDA: CUDA, @cuda
using DataFrames: DataFrame
using LoopVectorization: @avx

export ExpectationBit

function ExpectationBit(pc::ProbCircuit, lc::LogisticCircuit, data)
    pbc = ParamBitCircuitPair(pc, lc, data);
    ExpectationBit(pbc, pc, lc, data);
end

function ExpectationBit(pbc::ParamBitCircuitPair, pc::ProbCircuit, lc::LogisticCircuit, data)
    # 1. Get probability of each observation
    parambit = ParamBitCircuit(pbc.pc_bit, pbc.pc_params);
    log_likelihoods = MAR(parambit, data);
    p_observed = exp.( log_likelihoods )
    
    # 2. Expectation w.r.t. P(x_m, x_o)
    fvalues, gvalues = init_expectations(pbc, data, nothing, nothing, size(pbc.bcp.nodes)[2], num_classes(lc))
    expectation_layers(pbc, fvalues, gvalues)
    results_unnormalized = gvalues[:, end,:]

    # 3. Expectation w.r.t P(x_m | x_o)
    results = results_unnormalized ./ p_observed
    
    # # 4. Add Bias terms
    biases = lc.thetas
    results .+= biases
    
    results, fvalues, gvalues, pbc
end

function init_expectations(circuit::ParamBitCircuitPair, data, reuse_f, reuse_g, nodes_num, classes_num; Float=Float32)
    flowtype = isgpu(data) ? CuArray{Float} : Array{Float}
    fvalues = similar!(reuse_f, flowtype, num_examples(data), nodes_num)
    fgvalues = similar!(reuse_g, flowtype, num_examples(data), nodes_num, classes_num)

    # This is all only O(nfeatures) so was not worth doing parallization or on GPU
    data_cpu = to_cpu(data)
    nfeatures = num_features(data)

    fvalues[:,:] .= Float(0.0)   
    fgvalues[:,:,:] .= Float(0.0)  

    PARS = circuit.lc_params
    for var=1:nfeatures
        fvalues[.!isequal.(data_cpu[:, var], 0), var] .= Float(1.0)
        fvalues[.!isequal.(data_cpu[:, var], 1), var + 3*nfeatures] .= Float(1.0)

        
        # find the index of correct paramter to grab from LogisticCircuit
        p_ind = circuit.lc_bit.parents[circuit.lc_bit.nodes[3, 2+var]]
        p_ind2 = circuit.lc_bit.parents[circuit.lc_bit.nodes[3, 2+var+nfeatures]]

        for cc=1:classes_num
            fgvalues[:, var, cc] .= (fvalues[:, var] .* PARS[p_ind,cc])
            ind2 = var + 3*nfeatures
            fgvalues[:, ind2, cc] .= (fvalues[:, ind2] .* PARS[p_ind2,cc])
        end
    end
    return fvalues, fgvalues
end

function expectation_layers(circuit, fvalues, fgvalues)
    bcp::BitCircuitPair = circuit.bcp
    pc::BitCircuit = circuit.pc_bit
    lc::BitCircuit = circuit.lc_bit
    els = bcp.elements
    pc_pars = circuit.pc_params
    lc_pars = circuit.lc_params

    for layer in bcp.layers[2:end]
        Threads.@threads for dec_id in layer
            
            id_begin = @inbounds bcp.nodes[1, dec_id]
            id_end   = @inbounds bcp.nodes[2, dec_id]
            pc_id    = @inbounds bcp.nodes[5, dec_id]
            lc_id    = @inbounds bcp.nodes[6, dec_id]
            # println("!!, $dec_id,  $id_begin, $id_end, $pc_id, $lc_id")

            @inbounds pc_childs = 1 + pc.nodes[2, pc_id] - pc.nodes[1, pc_id]
            @inbounds lc_childs = 1 + lc.nodes[2, lc_id] - lc.nodes[1, lc_id]
            for A =  1 : pc_childs
                for B = 1 : lc_childs
                    shift = (A-1) * lc_childs + B
                    Z = id_begin + shift - 1
                    @inbounds PCZ = pc.nodes[1, pc_id] + A - 1
                    @inbounds LCZ = lc.nodes[1, lc_id] + B - 1

                    ## Compute fvalues[:, dec_id]
                    
                    @inbounds @fastmath @avx fvalues[:, dec_id] .+= exp(pc_pars[PCZ]) .* 
                        fvalues[:, els[2,Z]] .* 
                        fvalues[:, els[3,Z]]

                    ## Compute fgvalues[:, dec_id, :]
                    if els[3,Z] == els[2,Z]
                        # Special case (end in sum nodes)
                        @inbounds @fastmath @views @avx fgvalues[:, dec_id, :] .= 
                            transpose(lc_pars[LCZ,:]) .* 
                            fvalues[:, dec_id] 
                    else
                        @inbounds @fastmath @avx fgvalues[:, dec_id, :] .+=  
                            exp(pc_pars[PCZ]) .* 
                            transpose(lc_pars[LCZ,:]) .* 
                            (fvalues[:, els[2,Z]] .* fvalues[:, els[3,Z]])

                        @inbounds @fastmath @avx fgvalues[:, dec_id, :] .+= 
                            exp(pc_pars[PCZ]) .* 
                            ((fvalues[:, els[3,Z]] .* fgvalues[:, els[2,Z], :]) .+ 
                            (fvalues[:, els[2,Z]] .* fgvalues[:, els[3,Z], :]))
                    end
                    
                end # B
            end # A  

        end # dec_id
    end # layers
    
end