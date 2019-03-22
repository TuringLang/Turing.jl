##########################################
# Master file for running all test cases #
##########################################
using Turing; turnprogress(false)
using Pkg
using Test

# add packages
to_add = [
    PackageSpec(name="DynamicHMC"),
    PackageSpec(name="LogDensityProblems"),
]

Pkg.add(to_add)

# Import utility functions and reused models.
include("test_utils/utility.jl")
include("test_utils/models.jl")

@testset "Turing" begin
    @testset "core" begin
        include_dir("core/ad.jl")
        include_dir("core/compiler.jl")
        include_dir("core/container.jl")
        include_dir("core/VarReplay.jl")
    end

    @testset "inference" begin
        @testset "adapt" begin
            include_dir("inference/adapt/adapt.jl")
            include_dir("inference/adapt/precond.jl")
            include_dir("inference/adapt/stan.jl")
            include_dir("inference/adapt/stepsize.jl")
        end
        @testset "support" begin
            include_dir("inference/support/hmc_core.jl")
        end
        @testset "samplers" begin
            include_dir("inference/dynamichmc.jl")
            include_dir("inference/gibbs.jl")
            include_dir("inference/hmc.jl")
            include_dir("inference/hmcda.jl")
            include_dir("inference/ipmcmc.jl")
            include_dir("inference/is.jl")
            include_dir("inference/mh.jl")
            include_dir("inference/nuts.jl")
            include_dir("inference/pmmh.jl")
            include_dir("inference/sghmc.jl")
            include_dir("inference/sgld.jl")
            include_dir("inference/smc.jl")
        end
    end

    @testset "utilities" begin
        include_dir("utilities/distributions.jl")
        include_dir("utilities/io.jl")
        include_dir("utilities/resample.jl")
        include_dir("utilities/stan-interface.jl")
        include_dir("utilities/util.jl")
    end
end
