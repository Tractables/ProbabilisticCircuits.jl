using .BlossomV
using SparseArrays
using SimpleWeightedGraphs
using MetaGraphs: add_edge!
using ..Utils

export learn_vtree

#############
# Blossom bottom up method
# This method is only available if the user chooses to import BlossomV manually
#############

"Blossom bottom up method, vars are not used"
function blossom_bottom_up(data::DataFrame;α)
    weight = ones(Float64, num_examples(data))
    (_, mi) = mutual_information(data, weight; α)
    vars = Var.(collect(1:num_features(data)))
    info = round.(Int64, 1000001 .+ to_long_mi(mi, -1, -1000000))

    function f(leaf::Vector{<:Vtree})
        variable_sets = collect.(variables.(leaf))
        
        # even number of nodes, use blossomv alg
        function blossom_bottom_up_even!(variable_sets)::Tuple{Vector{Tuple{Var, Var}}, Int64}
            # 1. calculate pMI
            pMI = set_mutual_information(info, variable_sets)
            pMI = round.(Int64, pMI)

            # 2. solve by blossomv alg
            len = length(variable_sets)
            m = Matching(len)
            for i in 1 : len, j in i + 1 : len
                add_edge(m, i - 1, j - 1, pMI[i, j]) # blossomv index start from 0
            end

            solve(m)
            all_matches = Set{Tuple{Var, Var}}()
            for v in 1 : len
                push!(all_matches, order_asc(v, get_match(m, v - 1) + 1))
            end

            # 3. calculate scores, map index to var
            all_matches = Vector(collect(all_matches))
            score = 0

            for i in 1 : length(all_matches)
                (x, y) = all_matches[i]
                score += pMI[x, y]
            end

            return (all_matches, score)
        end

        # odd number of nodes, try every 2 combinations
        function blossom_bottom_up_odd!(variable_sets)
            # try all len - 1 conditions, find best score(minimun cost)
            (best_matches, best_score) = (nothing, typemax(Int64))
            len = length(variable_sets)
            for index in 1 : len
                indices = [collect(1:index-1);collect(index+1:len)]
                (matches, score) = blossom_bottom_up_even!(variable_sets[indices])
                if score < best_score
                    (best_matches, best_score) = ([[(indices[l], indices[r]) for (l,r) in matches];[index]], score)
                end
            end
            return (best_matches, best_score)
        end

        if length(variable_sets) % 2 == 0
            (matches, score) = blossom_bottom_up_even!(variable_sets)
        else
            (matches, score) = blossom_bottom_up_odd!(variable_sets)
        end
    
        pairs = []
        for x in matches
            if x isa Tuple
                push!(pairs, (leaf[x[1]], leaf[x[2]]))
            else
                push!(pairs, leaf[x])
            end
        end
        return pairs
    end
    return f
end
