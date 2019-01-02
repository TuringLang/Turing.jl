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

if get(ENV, "TRAVIS", "false") == "true"
    # If Travis is testing, separate the tests.
    numerical_tests = [joinpath("hmc.jl", "matrix_support.jl"),
                       joinpath("mh.jl", "mh_cons.jl"),
                       joinpath("models.jl", "single_dist_correctness.jl")]

    if ENV["STAGE"] == "test"
        runtests(exclude = numerical_tests)
    elseif ENV["STAGE"] == "numerical"
        runtests(specific_tests = numerical_tests)
    else
        @warn "Unknown Travis stage, currently set to: $(ENV["STAGE"])"
    end
else
    # Otherwise, test everything.
    runtests()
end

@debug("[runtests.jl] all tests finished")
