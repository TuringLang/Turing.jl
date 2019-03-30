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
        @testset "adapt" begin
            include("inference/adapt/adapt.jl")
            include("inference/adapt/precond.jl")
            include("inference/adapt/stan.jl")
            include("inference/adapt/stepsize.jl")
        end
        @testset "support" begin
            include("inference/support/hmc_core.jl")
        end
        @testset "samplers" begin
            include("inference/dynamichmc.jl")
            include("inference/gibbs.jl")
            include("inference/hmc.jl")
            include("inference/hmcda.jl")
            include("inference/ipmcmc.jl")
            include("inference/is.jl")
            include("inference/mh.jl")
            include("inference/nuts.jl")
            include("inference/pmmh.jl")
            include("inference/sghmc.jl")
            include("inference/sgld.jl")
            include("inference/smc.jl")
        end
    end

    @testset "utilities" begin
        include("utilities/distributions.jl")
        include("utilities/io.jl")
        include("utilities/resample.jl")
        include("utilities/stan-interface.jl")
        include("utilities/util.jl")
    end
end
