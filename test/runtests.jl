##########################################
# Master file for running all test cases #
##########################################
using Zygote, ReverseDiff, Memoization, Turing
using Pkg
using Random
using Test

setprogress!(false)

include("test_utils/AllUtils.jl")

# Begin testing.
@testset "Turing" begin
    @testset "core" begin
        include("core/ad.jl")
        include("core/container.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :tracker, :reversediff)
        Turing.setadbackend(adbackend)
        @testset "inference: $adbackend" begin
            @testset "samplers" begin
                include("inference/gibbs.jl")
                include("inference/hmc.jl")
                include("inference/is.jl")
                include("inference/mh.jl")
                include("inference/ess.jl")
                include("inference/emcee.jl")
                include("inference/AdvancedSMC.jl")
                include("inference/Inference.jl")
                include("contrib/inference/dynamichmc.jl")
            end
        end

        @testset "variational algorithms : $adbackend" begin
            include("variational/advi.jl")
        end

        @testset "modes" begin
            include("modes/ModeEstimation.jl")
        end
    end
    @testset "variational optimisers" begin
        include("variational/optimisers.jl")
    end

    Turing.setadbackend(:forwarddiff)
    @testset "stdlib" begin
        include("stdlib/distributions.jl")
        include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
        # include("utilities/stan-interface.jl")
        include("inference/utilities.jl")
    end
end
