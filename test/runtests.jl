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
        @timeit_include("inference/AdvancedSMC.jl")
        @timeit_include("inference/emcee.jl")
        @timeit_include("inference/ess.jl")
        @timeit_include("inference/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :reversediff)
        @timeit TIMEROUTPUT "inference: $adbackend" begin
            Turing.setadbackend(adbackend)
            @info "Testing $(adbackend)"
            @testset "inference: $adbackend" begin
                @testset "samplers" begin
                    @timeit_include("inference/gibbs.jl")
                    @timeit_include("inference/gibbs_conditional.jl")
                    @timeit_include("inference/hmc.jl")
                    @timeit_include("inference/Inference.jl")
                    @timeit_include("contrib/inference/dynamichmc.jl")
                    @timeit_include("contrib/inference/sghmc.jl")
                    @timeit_include("contrib/inference/abstractmcmc.jl")
                    @timeit_include("inference/mh.jl")
                end
            end

            @testset "variational algorithms : $adbackend" begin
                @timeit_include("variational/advi.jl")
            end

            @testset "modes : $adbackend" begin
                @timeit_include("modes/ModeEstimation.jl")
                @timeit_include("modes/OptimInterface.jl")
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
        @timeit_include("inference/utilities.jl")
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)
