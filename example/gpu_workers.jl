using Distributed, CUDA

# spawn one worker per device
if nprocs() - 1 < length(devices()) 
    addprocs(length(devices()) - nprocs() + 1)
end
@everywhere begin
    using Pkg; Pkg.activate("$(@__DIR__)")
    using CUDA
end

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end