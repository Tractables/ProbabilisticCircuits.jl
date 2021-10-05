export load_as_ensemble, save_as_ensemble

# TODO transition this to a Lerche.jl parser and use Base.{read,parse}

"Loads an ensemble from disk."
function load_as_ensemble(name::String; quiet::Bool = false)::Ensemble{StructProbCircuit}
    @assert endswith(name, ".esbl")
    zip = ZipFile.Reader(name)
    W, n = Vector{Float64}(), -1
    for f ∈ zip.files
        if endswith(f.name, ".meta")
            n = parse(Int, readline(f))
            W = map(x -> parse(Float64, x), split(readline(f)))
        end
    end
    @assert n > 0 && length(W) == n "Ensemble file format corrupted, empty or missing meta file."
    P = Tuple{Int, Int}[(0, 0) for i ∈ 1:n]
    for (i, f) ∈ enumerate(zip.files)
        if endswith(f.name, ".psdd")
            j = parse(Int, f.name[1:end-5])
            @assert j > 0 && j <= n "Either .meta file is corrupted or .psdd is misnamed (faulty: $(f.name))."
            P[j] = (i, P[j][2])
        elseif endswith(f.name, ".vtree")
            j = parse(Int, f.name[1:end-6])
            @assert j > 0 && j <= n "Either .meta file is corrupted or .vtree is misnamed (faulty: $(f.name))."
            P[j] = (P[j][1], i)
        end
    end
    C = Vector{StructProbCircuit}(undef, n)
    function do_work(k::Int, i::Int, j::Int)
        @assert i > 0 "Missing .psdd file for the $k-th circuit."
        @assert j > 0 "Missing .psdd file for the $k-th circuit."
        psdd_file, vtree_file = zip.files[i], zip.files[j]
        psdd, _ = load_struct_prob_circuit(psdd_file, vtree_file)
        C[k] = psdd
        nothing
    end
    !quiet && print("Loading circuits...\n  ")
    for (k, (i, j)) ∈ enumerate(P)
        do_work(k, i, j)
        !quiet && print('*')
    end
    !quiet && print('\n')
    close(zip)
    return Ensemble{StructProbCircuit}(C, W)
end


"Save file as a .esbl ensemble file format."
function save_as_ensemble(name::String, ensemble::Ensemble{StructProbCircuit}; quiet::Bool = false)
    @assert endswith(name, ".esbl")
    zip = ZipFile.Writer(name)
    f_w = ZipFile.addfile(zip, "ensemble.meta")
    n = length(ensemble.C)
    write(f_w, "$(n)\n")
    write(f_w, join(ensemble.W, ' '))
    close(f_w)
    function do_work(C::StructProbCircuit, i::Integer)
        f_c = ZipFile.addfile(zip, "$(i).psdd")
        save_as_psdd(f_c, C, C.vtree)
        f_v = ZipFile.addfile(zip, "$(i).vtree")
        save_vtree(f_v, C.vtree)
        nothing
    end
    !quiet && print("Saving circuits...\n  ")
    for (i, C) ∈ enumerate(ensemble.C)
        do_work(C, i)
        !quiet && print('*')
    end
    !quiet && print('\n')
    close(zip)
    nothing
end
