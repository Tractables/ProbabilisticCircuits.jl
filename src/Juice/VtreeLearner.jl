using BlossomV
using Metis
using SparseArrays

const δINT = 1000000
const MIN_INT = 1
const MAX_INT = δINT + MIN_INT

function to_long_mi(m::Matrix{Float64})::Matrix{Int64}
    δmi = maximum(m) - minimum(m)
    return @. round(Int64, m * δINT / δmi + MIN_INT)
end

@inline order_asc(x, y) = x > y ? (y, x) : (x , y)

#############
# Metis top down method
#############

struct MetisContext <: VtreeLearnerContext
    info::Matrix{Int64}
end

MetisContext(mi::Matrix{Float64}) = MetisContext(to_long_mi(mi))

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
# Blossom bottom up method
#############

mutable struct BlossomContext <: VtreeLearnerContext
    variable_sets::Vector{Vector{Var}}
    partition_id::Vector{Int64} # map vars to index in variable_sets
    info::Matrix{Int64}
end

BlossomContext(vars::Set{Var}, mi::Matrix{Float64}) =
    BlossomContext( [[v] for v in sort(collect(vars))],
                    collect(1 : length(vars)),
                    MAX_INT .+ MIN_INT .- to_long_mi(mi))

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

        "3. calculate scores, map index to var"
        all_matches = Vector(collect(all_matches))
        score = 0
        for i in 1 : length(all_matches)
            (x, y) = all_matches[i]
            score += pMI[x, y]
            all_matches[i] = (context.variable_sets[x][1], context.variable_sets[y][1])
        end
        all_matches = Set(all_matches)

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
        (best_index, best_matches, best_score) = (0, Set{Tuple{Var, Var}}(), typemax(Int64))

        for index in 1 : length(context.variable_sets)
            set = copy(sub_context.variable_sets[index])
            deleteat!(sub_context.variable_sets, index)

            (matches, score) = blossom_bottom_up_even!(vars, sub_context; update = false)
            if score < best_score
                (best_index, best_matches, best_score) = (index, matches, score)
            end

            insert!(sub_context.variable_sets, index, set)
        end

        "2. update information"
        updata_context(best_matches, context)
        return (best_matches, best_score)

    end

    function updata_context(matches::Set{Tuple{Var, Var}}, context::BlossomContext)
        for (x, y) in matches
            y_partition = copy(context.variable_sets[context.partition_id[y]])
            context.variable_sets[context.partition_id[y]] = Vector()
            foreach(value -> push!(context.variable_sets[context.partition_id[x]], value), y_partition)
        end

        context.variable_sets = [x for x in context.variable_sets if x != []]
        for index in 1 : length(context.variable_sets)
            for y in context.variable_sets[index]
                context.partition_id[y] = index
            end
        end
    end

    if length(vars) % 2 == 0
        (matches, score) = blossom_bottom_up_even!(vars, context)
    else
        (matches, score) = blossom_bottom_up_odd!(vars, context)
    end

    for (left, right) in matches
        pop!(vars, right)
    end

    return matches
end


#############
# Test method
#############

"Test context, learn vtree by stipulated method"
struct TestContext <: VtreeLearnerContext
end

"Test top down method, split nodes by ascending order, balanced"
function test_top_down(vars::Set{Var}, context::TestContext)::Tuple{Set{Var}, Set{Var}}
    sorted_vars = sort(collect(vars))
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
