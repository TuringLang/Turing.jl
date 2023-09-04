using Turing; include(pkgdir(Turing)*"/test/test_utils/test_utils.jl")

setprogress!(false)

# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
macro timeit_include(path::AbstractString) :(@timeit TIMEROUTPUT $path include($path)) end

@testset "Turing" begin
    @testset "essential" begin
        @timeit_include("essential/ad.jl")
    end

    @testset "samplers (without AD)" begin
        @timeit_include("mcmc/particle_mcmc.jl")
        @timeit_include("mcmc/emcee.jl")
        @timeit_include("mcmc/ess.jl")
        @timeit_include("mcmc/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :reversediff)
        @timeit TIMEROUTPUT "inference: $adbackend" begin
            Turing.setadbackend(adbackend)
            @info "Testing $(adbackend)"
            @testset "inference: $adbackend" begin
                @testset "samplers" begin
                    @timeit_include("mcmc/gibbs.jl")
                    @timeit_include("mcmc/gibbs_conditional.jl")
                    @timeit_include("mcmc/hmc.jl")
                    @timeit_include("mcmc/Inference.jl")
                    @timeit_include("mcmc/sghmc.jl")
                    @timeit_include("mcmc/abstractmcmc.jl")
                    @timeit_include("mcmc/mh.jl")
                    @timeit_include("ext/dynamichmc.jl")
                end
            end

            @testset "variational algorithms : $adbackend" begin
                @timeit_include("variational/advi.jl")
            end

            @testset "mode estimation : $adbackend" begin
                @timeit_include("optimisation/OptimInterface.jl")
                @timeit_include("ext/Optimisation.jl")
            end

        end
    end

    @testset "variational optimisers" begin
        @timeit_include("variational/optimisers.jl")
    end

    Turing.setadbackend(:forwarddiff)
    @testset "stdlib" begin
        @timeit_include("stdlib/distributions.jl")
        @timeit_include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
        @timeit_include("mcmc/utilities.jl")
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)
