using AbstractMCMC
using AdvancedMH
using Clustering
using Distributions
using Distributions.FillArrays
using DistributionsAD
using FiniteDifferences
using ForwardDiff
using GalacticOptim
using MCMCChains
using Memoization
using NamedArrays
using Optim
using PDMats
using ReverseDiff
using SpecialFunctions
using StatsBase
using StatsFuns
using Tracker
using Turing
using Turing.Inference
using Turing.RandomMeasures
using Zygote

using LinearAlgebra
using Pkg
using Random
using Test

using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using AdvancedVI: TruncatedADAGrad, DecayedADAGrad, apply!
using Distributions: Binomial, logpdf
using DynamicPPL: getval, getlogp
using ForwardDiff: Dual
using MCMCChains: Chains
using StatsFuns: binomlogpdf, logistic, logsumexp
using Turing: BinomialLogit, ForwardDiffAD, Sampler, SampleFromPrior, NUTS, TrackerAD,
                Variational, ZygoteAD, getspace, gradient_logp
using Turing.Essential: TuringDenseMvNormal, TuringDiagMvNormal
using Turing.Variational: TruncatedADAGrad, DecayedADAGrad, AdvancedVI

setprogress!(false)

include(pkgdir(Turing)*"/test/test_utils/AllUtils.jl")

@testset "Turing" begin
    @testset "essential" begin
        include("essential/ad.jl")
    end

    @testset "samplers (without AD)" begin
        include("inference/AdvancedSMC.jl")
        include("inference/emcee.jl")
        include("inference/ess.jl")
        include("inference/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :tracker, :reversediff)
        Turing.setadbackend(adbackend)
        @info "Testing $(adbackend)"
        start = time()
        @testset "inference: $adbackend" begin
            @testset "samplers" begin
                include("inference/gibbs.jl")
                include("inference/gibbs_conditional.jl")
                include("inference/hmc.jl")
                include("inference/Inference.jl")
                include("contrib/inference/dynamichmc.jl")
                include("contrib/inference/sghmc.jl")
                include("inference/mh.jl")
            end
        end

        @testset "variational algorithms : $adbackend" begin
            include("variational/advi.jl")
        end

        @testset "modes" begin
            include("modes/ModeEstimation.jl")
            include("modes/OptimInterface.jl")
        end

        # Useful for
        # a) discovering performance regressions,
        # b) figuring out why CI is timing out.
        @info "Tests for $(adbackend) took $(time() - start) seconds"
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
