using Test;

include("../src/Juice/Juice.jl")
using .Juice

tests = [
   "CircuitParserTest.jl",
   "CircuitQueriesTest.jl",
   "MixtureCircuitsTest.jl",
   "LogicTest.jl",
   "VtreeParserTest.jl"
   ]

@testset "Juice" begin
   for test in tests
     include(test)
   end
end
