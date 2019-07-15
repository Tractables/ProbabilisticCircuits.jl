using Test;

include("../src/Juice/Juice.jl")
using .Juice

@testset "Juice" begin
   include("CircuitParserTest.jl")
   include("CircuitQueriesTest.jl")
   include("ChowLiuTreeTest.jl")
end
