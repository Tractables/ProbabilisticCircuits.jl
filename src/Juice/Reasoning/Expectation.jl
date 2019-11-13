ExpCacheDict = Dict{Pair{ProbΔNode, LogisticΔNode}, Array{Float64, 2}}

struct ExpectationCache 
    f::ExpCacheDict
    fg::ExpCacheDict
end

ExpectationCache() = ExpectationCache(ExpCacheDict(), ExpCacheDict())

# On Tractable Computation of Expected Predictions (https://arxiv.org/abs/1910.02182)
"""
Missing values should be denoted by -1
"""
function Expectation(pc::ProbΔ, lc::LogisticΔ, data::XData{Int8})
    # 1. Get probability of each observation
    fc, log_likelihoods = marginal_log_likelihood_per_instance(pc, data)
    p_observed = exp.( log_likelihoods )
    
    # 2. Expectation w.r.t. P(x_m, x_o)
    cache = ExpectationCache()
    results_unnormalized = exp_g(pc[end], lc[end-1], data, cache) # skipping the bias node of lc

    # 3. Expectation w.r.t P(x_m | x_o)
    results = transpose(results_unnormalized) ./ p_observed

    # 4. Add Bias terms
    biases = lc[end].thetas
    results .+= biases
    
    results, cache
end

function ExpectationUpward(pc::ProbΔ, lc::LogisticΔ, data::XData{Int8})
        # 1. Get probability of each observation
        fc, log_likelihoods = marginal_log_likelihood_per_instance(pc, data)
        p_observed = exp.( log_likelihoods )
        
        # 2. Expectation w.r.t. P(x_m, x_o)
        exps_flow = exp_pass_up(pc, lc, data)
        results_unnormalized = exps_flow[end].fg
    
        # 3. Expectation w.r.t P(x_m | x_o)
        results = transpose(results_unnormalized) ./ p_observed
    
        # 4. Add Bias terms
        biases = lc[end].thetas
        results .+= biases
        
        results, exps_flow
end


# exp_f (pr-constraint) is originally from:
#   Arthur Choi, Guy Van den Broeck, and Adnan Darwiche. Tractable learning for structured probability spaces: A case study in learning preference distributions. In Proceedings of IJCAI, 2015.

function exp_f(n::Prob⋁, m::Logistic⋁, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.f, Pair(n, m)) do
        value = zeros(1 , num_examples(data) )
        pthetas = [exp(n.log_thetas[i]) for i in 1:length(n.children)]
        @fastmath @simd for i in 1:length(n.children)
            @simd for j in 1:length(m.children)
                value .+= (pthetas[i] .* exp_f(n.children[i], m.children[j], data, cache))
            end
        end
        return value
    end
end

function exp_f(n::Prob⋀, m::Logistic⋀, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.f, Pair(n, m)) do
        value = ones(1 , num_examples(data) )
        @fastmath for (i,j) in zip(n.children, m.children)
            value .*= exp_f(i, j, data, cache)
        end
        return value
        # exp_f(n.children[1], m.children[1], data, cache) .* exp_f(n.children[2], m.children[2], data, cache)
    end
end


@inline function exp_f(n::ProbLiteral, m::LogisticLiteral, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.f, Pair(n, m)) do
        value = zeros(1 , num_examples(data) )
        var = lit2var(literal(m))
        X = feature_matrix(data)
        if positive(n) && positive(m) 
            # value[1, X[:, var] .== -1 ] .= 1.0  # missing observation always agrees
            # value[1, X[:, var] .== 1 ] .= 1.0 # positive observations
            value[1, X[:, var] .!= 0 ] .= 1.0 # positive or missing observations
        elseif negative(n) && negative(m)
            # value[1, X[:, var] .== -1 ] .= 1.0  # missing observation always agrees
            # value[1, X[:, var] .== 0 ] .= 1.0 # negative observations
            value[1, X[:, var] .!= 1 ] .= 1.0 # negative or missing observations
        end
        return value
    end
end

"""
Has to be a Logistic⋁ with only one child, which is a leaf node 
"""
@inline function exp_f(n::ProbLiteral, m::Logistic⋁, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.f, Pair(n, m)) do
        exp_f(n, m.children[1], data, cache)
    end
end

@inline function exp_g(n::Prob⋁, m::Logistic⋁, data::XData{Int8}, cache::ExpectationCache)
    exp_fg(n, m, data, cache) # exp_fg and exp_g are the same for OR nodes
end

# function exp_g(n::Prob⋀, m::Logistic⋀, data::XData{Int8}, cache::ExpectationCache)
#     value = zeros(classes(m) , num_examples(data))
#     @fastmath for (i,j) in zip(n.children, m.children)
#         value .+= exp_fg(i, j, data, cache)
#     end
#     return value
#     # exp_fg(n.children[1], m.children[1], data, cache) .+ exp_fg(n.children[2], m.children[2], data, cache)
# end


function exp_fg(n::Prob⋁, m::Logistic⋁, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.fg, Pair(n, m)) do
        value = zeros(classes(m) , num_examples(data) )
        pthetas = [exp(n.log_thetas[i]) for i in 1:length(n.children)]
        @fastmath @simd for i in 1:length(n.children)
            for j in 1:length(m.children)
                value .+= (pthetas[i] .* m.thetas[j,:]) .* exp_f(n.children[i], m.children[j], data, cache)
                value .+= pthetas[i] .* exp_fg(n.children[i], m.children[j], data, cache)
            end
        end
        return value
    end
end

function exp_fg(n::Prob⋀, m::Logistic⋀, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.fg, Pair(n, m)) do
        # Assuming 2 children
        value = exp_f(n.children[1], m.children[1], data, cache) .* exp_fg(n.children[2], m.children[2], data, cache)
        value .+= exp_f(n.children[2], m.children[2], data, cache) .* exp_fg(n.children[1], m.children[1], data, cache)
        return value
    end
end


"""
Has to be a Logistic⋁ with only one child, which is a leaf node 
"""
@inline function exp_fg(n::ProbLiteral, m::Logistic⋁, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.fg, Pair(n, m)) do
        m.thetas[1,:] .* exp_f(n, m, data, cache)
    end
end

@inline function exp_fg(n::ProbLiteral, m::LogisticLiteral, data::XData{Int8}, cache::ExpectationCache)
    @inbounds get!(cache.fg, Pair(n, m)) do
        exp_f(n, m, data, cache)
    end
end