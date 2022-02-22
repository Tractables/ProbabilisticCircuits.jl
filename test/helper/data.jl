"""
Generates all possible binary configurations of size N
"""
function generate_data_all(N::Int)
    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all,
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    Matrix(data_all)
end