##########################################
# Master file for running all test cases #
##########################################
using Turing; turnprogress(false)
using Pkg
using Random
using Test

include("test_utils/AllUtils.jl")

# Begin testing.
@testset "Turing" begin
    @testset "core" begin
        include("core/ad.jl")
        include("core/compiler.jl")
        include("core/container.jl")
        include("core/RandomVariables.jl")
    end

    @testset "inference" begin
        @testset "samplers" begin
            # DynamicHMC version 1 has a bug on 32bit pllatforms
            if Int === Int64 && Pkg.installed()["DynamicHMC"] < v"2"
                include("contrib/inference/dynamichmc.jl")
            end
            include("inference/gibbs.jl")
            include("inference/hmc.jl")
            include("inference/is.jl")
            include("inference/mh.jl")
            include("inference/AdvancedSMC.jl")
            include("inference/Inference.jl")
        end
    end

    @testset "variational" begin
        @testset "algorithms" begin
            include("variational/advi.jl")
        end
    end

    @testset "stdlib" begin
        include("stdlib/distributions.jl")
        # include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
      # include("utilities/stan-interface.jl")
        include("utilities/util.jl")
    end
end
