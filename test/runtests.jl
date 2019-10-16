if endswith(@__FILE__, PROGRAM_FILE)
   # this file is run as a script
   include("../src/Juice/Juice.jl")
end

using Test

@testset "Juice-All" begin
   for test in [
      "Utils/runtests.jl",
      "Data/runtests.jl",
      "Juice/runtests.jl"
      ]
     @info "Starting tests for {$test}"
     include(test)
   end
end
