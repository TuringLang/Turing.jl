##########################################
# Master file for running all test cases #
##########################################
using Zygote, ReverseDiff, Turing; turnprogress(false)
using Pkg
using Random
using Test

include("test_utils/AllUtils.jl")

# Begin testing.
@testset "Turing" begin
    @testset "core" begin
        include("core/ad.jl")
        include("core/container.jl")
    end

    test_adbackends = if VERSION >= v"1.2"
        [:forwarddiff, :tracker, :reversediff]
    else
        [:forwarddiff, :tracker]
    end
    for adbackend in test_adbackends
        Turing.setadbackend(adbackend)
        @testset "inference" begin
            @testset "samplers" begin
                # FIXME: DynamicHMC version 1 has (??) a bug on 32bit platforms (but we were too
                # lazy to open an issue so Tamas doesn't know about it), retest with 2.0
                if Int === Int64 && Pkg.installed()["DynamicHMC"].major == 2
                    include("contrib/inference/dynamichmc.jl")
                end
                include("inference/gibbs.jl")
                include("inference/hmc.jl")
                include("inference/is.jl")
                include("inference/mh.jl")
                include("inference/ess.jl")
                include("inference/AdvancedSMC.jl")
                include("inference/Inference.jl")
            end
        end

        @testset "variational" begin
            @testset "algorithms" begin
                include("variational/advi.jl")
            end

            @testset "optimisers" begin
                include("variational/optimisers.jl")
            end
        end
    end

    Turing.setadbackend(:forwarddiff)
    @testset "stdlib" begin
        include("stdlib/distributions.jl")
        # include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
      # include("utilities/stan-interface.jl")
        include("utilities/util.jl")
    end
end
