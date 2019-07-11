[![Build Status](https://travis-ci.com/UCLA-StarAI/Juice.jl.svg?token=WqP1S31vh9msACoVUepf&branch=master)](https://travis-ci.com/UCLA-StarAI/Juice.jl)

# Juice.jl
Julia Circuit Empanada


# Requirements

Run the following to install the dependencies.

 ``julia install.jl``


# Developement Tips

## Using Revise

`Revise.jl` allows you to modify code and use the changes without restarting Julia. For more information refer to [their github repository](https://github.com/timholy/Revise.jl). 

## CircuitSandbox

`CircuitsSandbox.jl` will contain some examples on how to use the library. To use the Revise functionality, run the following command at the beginning of your REPL session.

    using Revise; track();

## Testing

We will be adding automated tests under `/test` folder. At the moment, the following tests are available:

    julia -L test/CircuitParserTest.jl

Right now the tests are not stable yet, but they become stable make sure to run the tests before commiting new code.
