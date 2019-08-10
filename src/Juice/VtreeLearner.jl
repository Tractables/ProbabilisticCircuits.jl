using BlossomV
using Metis
using SparseArrays

const LONG_TIMES = 10000000
@inline to_long(m::Matrix{Float64})::Matrix{Int64} = trunc.(Int64, m * LONG_TIMES)


#############
# Metis top down method
#############

struct MetisContext <: VtreeLearnerContext
    info::Matrix{Int64}
end

MetisContext(mi::Matrix{Float64}) = MetisContext(to_long(mi))

"Metis top down method"
function metis_top_down(vars::Set{Var}, context::MetisContext)::Tuple{Set{Var}, Set{Var}}

    vertices = sort(collect(vars))
    sub_context = context.info[vertices, vertices]
    graph = convert(SparseMatrixCSC, sub_context)
    partition = Metis.partition(graph, 2, alg = :RECURSIVE)

    subsets = (Set{Var}(), Set{Var}())
    for (index, p) in enumerate(partition)
        push!(subsets[p], vertices[index])
    end

    return subsets
end


#############
# Bottom up method
#############

mutable struct BlossomContext <: VtreeLearnerContext
    variable_sets::Vector{Vector{Var}}
    partition_id::Vector{Int64} # map vars to index in variable_sets
    info::Matrix{Int64}
end

BlossomContext(vars::Set{Var}, mi::Matrix{Float64}) =
    BlossomContext( [[v] for v in sort(collect(vars))],
                    collect(1 : len(vars)), - to_long(mi))

"Blossom bottom up method, vars are not used"
function blossom_bottom_up!(vars::Set{Var}, context::BlossomContext)::Set{Tuple{Var, Var}}

    "even number of nodes, use blossomv alg"
    function blossom_bottom_up_even!(vars::Set{Var}, context::BlossomContext; update = true)::Tuple{Set{Tuple{Var, Var}}, Int64}
        "1. calculate pMI"
        pMI = set_mi(context.info, context.variable_sets)

        "2. solve by blossomv alg"
        len = length(context.variable_sets)
        m = Matching(len)
        for i in 1 : len, j in i + 1 : len
            add_edge(m, i - 1, j - 1, pMI[i, j]) # blossomv index start from 0
        end

        solve(m)
        all_matches = Set{Tuple{Var, Var}}()
        for v in 1 : len
            push!(all_matches, order_asc(v, get_match(m, v - 1) + 1))
        end

        "3. calculate scores"
        score = 0
        for (x, y) in all_matches
            score += pMI[x, y]
        end

        "4. update context when called by outer layer"
        if update
            updata_context(all_matches, context)
        end

        return (all_matches, score)
    end

    "odd number of nodes, try every 2 combinations"
    function blossom_bottom_up_odd!(vars::Set{Var}, context::BlossomContext)::Tuple{Set{Tuple{Var, Var}}, Int64}
        sub_context = deepcopy(context)

        "1. try all len - 1 conditions, find best score(minimun cost)"
        best_matches =  Set{Tuple{Var, Var}}()
        best_score = typemax(Int64)
        best_index = 0

        for index in 1 : length(context.variable_sets)
            set = deleteat!(sub_context.variable_sets, index)

            (matches, score) = blossom_bottom_up_even!(vars, sub_context; update = false)
            if score < best_score
                best_matches = matches
                best_score = score
                best_index = index
            end

            insert!(sub_context.variables, index, set)
        end

        "2. update information"
        updata_context(best_matches, context)
        return (best_matches, best_score)

    end

    function updata_context(matches::Set{Tuple{Var, Var}}, context::BlossomContext)
        for (x, y) in matches
            y_partition = pop!(context.variable_sets[partition_id[y]])
            context.variable_sets[partition_id[x]] += y_partition
            for tmp in y_partition
                partition_id[tmp] = partition_id[x]
            end
        end
    end

    if length(vars) % 2 == 0
        (matches, score) = blossom_bottom_up_even!(vars, context)
    else
        (matches, score) = blossom_bottom_up_odd!(vars, context)
    end
    
    return matches
end


#############
# Test method
#############

"Test context, learn vtree by stipulated method"
mutable struct TestContext <: VtreeLearnerContext
    variables::Set{Var}
end

"Test top down method, split nodes by ascending order, balanced"
function test_top_down(vars::Set{Var}, context::TestContext)::Tuple{Set{Var}, Set{Var}}
    context.variables = vars
    sorted_vars = sort(collect(context.variables))
    len = length(sorted_vars)
    len1 = Int64(len % 2 == 0 ? len // 2 : (len - 1) // 2)
    return (Set(sorted_vars[1 : len1]), Set(sorted_vars[len1 + 1 : end]))
end

"Test bottom up method, split nodes by ascending order, balanced"
function test_bottom_up!(vars::Set{Var}, context::TestContext)::Set{Tuple{Var, Var}}
    sorted_vars = sort(collect(vars))
    len = length(sorted_vars)
    len1 = Int64(len % 2 == 0 ? len // 2 : (len - 1) // 2)
    matches = Set{Tuple{Var, Var}}()
    for i in 1 : len1
        push!(matches, (sorted_vars[2 * i - 1], sorted_vars[2 * i]))
        pop!(vars, sorted_vars[2 * i])
    end
    return matches
end
