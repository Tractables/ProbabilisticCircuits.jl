using Test;

include("../src/Juice/Juice.jl")
using .Juice

tests = [
   "CircuitParserTest.jl",
   "CircuitQueriesTest.jl",
   "ChowLiuTreeTest.jl",
   "CircuitBuilderTest.jl",
   "LogicTest.jl"
   ]

@testset "Juice" begin
   for test in tests
     include(test)
   end
end
