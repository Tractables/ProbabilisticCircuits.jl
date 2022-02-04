using CUDA, Random

function debug(num_examples)
    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    
    prefix = @view shuffled_indices[1:num_examples÷2]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    i = 1
    while true
        do_shuffle()
        if i%3000 == 0
            println("$(i) - $(sum(prefix))")
        end
        i+=1
    end
end

debug(50000)