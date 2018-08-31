##########################################
# Master file for running all test cases #
##########################################
using Turing; turnprogress(false)
using Test

@debug("[runtests.jl] runtests.jl loaded")

path = dirname(@__FILE__)
cd(path); include("utility.jl")
@debug("[runtests.jl] utility.jl loaded")
@debug("[runtests.jl] testing starts")

# run all tests
runtests()

@debug("[runtests.jl] all tests finished")
