using CUDA: CUDA

function cpu_gpu_agree(f, data; atol=1e-7)
    CUDA.functional() && @test f(data) == to_cpu(f(to_gpu(data)))
end

function cpu_gpu_agree_approx(f, data; atol=1e-7)
    CUDA.functional() && @test f(data) ≈ to_cpu(f(to_gpu(data))) atol=atol
end