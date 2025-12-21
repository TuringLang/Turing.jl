include("test_utils/SelectiveTests.jl")
using .SelectiveTests: isincluded, parse_args
using Pkg
using Random: seed!
using Test
using TimerOutputs: TimerOutputs, @timeit
import Turing

# Fix the global Random.seed for reproducibility.
seed!(23)

include("test_utils/models.jl")
include("test_utils/numerical_tests.jl")
include("test_utils/sampler.jl")

Turing.setprogress!(false)
included_paths, excluded_paths = parse_args(ARGS)

# Filter which tests to run and collect timing and allocations information to show in a
# clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
macro timeit_include(path::AbstractString)
    return quote
        if isincluded($path, included_paths, excluded_paths)
            @timeit TIMEROUTPUT $path include($path)
        else
            println("Skipping tests in $($path)")
        end
    end
end

@testset "Turing" verbose = true begin
    @testset "Aqua" begin
        @timeit_include("Aqua.jl")
    end

    @testset "AD" verbose = true begin
        @timeit_include("ad.jl")
    end

    @testset "essential" verbose = true begin
        @timeit_include("essential/container.jl")
    end

    @testset "samplers (without AD)" verbose = true begin
        @timeit_include("mcmc/abstractmcmc.jl")
        @timeit_include("mcmc/particle_mcmc.jl")
        @timeit_include("mcmc/emcee.jl")
        @timeit_include("mcmc/ess.jl")
        @timeit_include("mcmc/is.jl")
    end

    @timeit TIMEROUTPUT "inference" begin
        @testset "inference with samplers" verbose = true begin
            @timeit_include("mcmc/gibbs.jl")
            @timeit_include("mcmc/gibbs_conditional.jl")
            @timeit_include("mcmc/hmc.jl")
		@timeit_include("mcmc/initial_params.jl")
            @timeit_include("mcmc/Inference.jl")
            @timeit_include("mcmc/sghmc.jl")
            @timeit_include("mcmc/external_sampler.jl")
            @timeit_include("mcmc/mh.jl")
            @timeit_include("ext/dynamichmc.jl")
            @timeit_include("mcmc/repeat_sampler.jl")
        end

        @testset "variational algorithms" begin
            @timeit_include("variational/vi.jl")
        end

        @testset "mode estimation" verbose = true begin
            @timeit_include("optimisation/Optimisation.jl")
        end
    end

    @testset "stdlib" verbose = true begin
        @timeit_include("stdlib/distributions.jl")
        @timeit_include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
        @timeit_include("mcmc/utilities.jl")
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)
