[![Build Status](https://travis-ci.com/UCLA-StarAI/Juice.jl.svg?token=WqP1S31vh9msACoVUepf&branch=master)](https://travis-ci.com/UCLA-StarAI/Juice.jl)

# Juice.jl
Julia Circuit Empanada


# Requirements

Assuming you have Julia 1.1 installed, run the following to install the dependencies.

 ``julia install.jl``


# Documentation

To build the documentation locally, run the following to build the documentation, and then open `docs/build/index.html`.

    julia docs/make.jl




# Developement Tips

## Using Revise

`Revise.jl` allows you to modify code and use the changes without restarting Julia. For more information refer to [their github repository](https://github.com/timholy/Revise.jl).
 To use the Revise functionality, run the following command at the beginning of your REPL session.

    using Revise; include("test/Debug.jl");


## Sandbox

`Sandbox.jl` will contain some examples on how to use the library.

## Testing

We will be adding automated tests under `/test` folder. To run all the test cases, run:

    julia test/runtests.jl

To run an individual test you can use `/test/runtest.jl`, for example:

    julia test/runtest.jl CircuitParserTest.jl
    julia test/runtest.jl CircuitQueriesTest.jl
    julia test/runtest.jl MixtureCircuitsTest.jl
    julia test/runtest.jl LogicTest.jl
    julia test/runtest.jl VtreeParserTest.jl

Make sure to run the tests before commiting new code.
