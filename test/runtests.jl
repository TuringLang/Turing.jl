using AbstractMCMC
using AdvancedMH
using Clustering
using Distributions
using FiniteDifferences
using ForwardDiff
using MCMCChains
using Memoization
using Random
using ReverseDiff
using PDMats
using StatsFuns
using Tracker
using Turing
using Turing.Inference
using Turing.RandomMeasures
using Zygote

# Julia base.
using LinearAlgebra
using Pkg
using Test

using DynamicPPL: getval, getlogp
using ForwardDiff: Dual
using MCMCChains: Chains
using StatsFuns: binomlogpdf, logistic, logsumexp
using Turing: Sampler, SampleFromPrior, NUTS, TrackerAD, ZygoteAD, getspace
using Turing.Core: TuringDenseMvNormal, TuringDiagMvNormal

setprogress!(false)

include("test_utils/AllUtils.jl")

@testset "Turing" begin
    @testset "core" begin
        include("core/ad.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :tracker, :reversediff)
        Turing.setadbackend(adbackend)
        @testset "inference: $adbackend" begin
            @testset "samplers" begin
                include("inference/gibbs.jl")
                include("inference/gibbs_conditional.jl")
                include("inference/hmc.jl")
                include("inference/is.jl")
                include("inference/mh.jl")
                include("inference/ess.jl")
                include("inference/emcee.jl")
                include("inference/AdvancedSMC.jl")
                include("inference/Inference.jl")
                include("contrib/inference/dynamichmc.jl")
                include("contrib/inference/sghmc.jl")
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
