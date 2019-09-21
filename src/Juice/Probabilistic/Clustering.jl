using Clustering

function clustering(train_x::XData, mix_num::Int64; maxiter=200)::XBatches{<:Bool}
    if mix_num == 1
        return convert(XBatches, train_x)
    end
    
    n = num_examples(train_x)
    X = feature_matrix(train_x)'

    println("Running K-means clustering algorithm with num of components $mix_num, maximum iterations $maxiter")
    @time R = kmeans(X, mix_num; maxiter=maxiter)
    @assert nclusters(R) == mix_num
    a = assignments(R)

    clustered_train_x = Vector{PlainXData{Bool,BitMatrix}}()
    for k in 1 : mix_num
        push!(clustered_train_x, XData(convert(BitMatrix, X[:, findall(x -> x == k, a)]')))
    end
    return clustered_train_x
end