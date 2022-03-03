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
using OrderedCollections
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
using StableRNGs

using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using AdvancedVI: TruncatedADAGrad, DecayedADAGrad, apply!
using DataFrames: DataFrame
using Distributions: Binomial, logpdf
using DynamicPPL: getval, getlogp
using ForwardDiff: Dual
using MCMCChains: Chains
using OrderedCollections: OrderedDict
using StatsFuns: binomlogpdf, logistic, logsumexp
using Turing: BinomialLogit, ForwardDiffAD, Sampler, SampleFromPrior, NUTS, TrackerAD,
                Variational, ZygoteAD, getspace, gradient_logp
using Turing.Essential: TuringDenseMvNormal, TuringDiagMvNormal
using Turing.Variational: TruncatedADAGrad, DecayedADAGrad, AdvancedVI

setprogress!(false)

include(pkgdir(Turing)*"/test/test_utils/AllUtils.jl")

# Collect timing and allocations information to show in a clear way.
include(pkgdir(Turing)*"/test/test_utils/timing.jl")

@testset "Turing" begin
    @testset "essential" begin
        time_include("essential/ad.jl")
    end

    @testset "samplers (without AD)" begin
        time_include("inference/AdvancedSMC.jl")
        time_include("inference/emcee.jl")
        time_include("inference/ess.jl")
        time_include("inference/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :tracker, :reversediff)
        Turing.setadbackend(adbackend)
        @info "Testing $(adbackend)"
        start = time()
        @testset "inference: $adbackend" begin
            @testset "samplers" begin
                time_include("inference/gibbs.jl", adbackend)
                time_include("inference/gibbs_conditional.jl", adbackend)
                time_include("inference/hmc.jl", adbackend)
                time_include("inference/Inference.jl", adbackend)
                time_include("contrib/inference/dynamichmc.jl", adbackend)
                time_include("contrib/inference/sghmc.jl", adbackend)
                time_include("inference/mh.jl", adbackend)
            end
        end

        @testset "variational algorithms : $adbackend" begin
            time_include("variational/advi.jl", adbackend)
        end

        @testset "modes" begin
            time_include("modes/ModeEstimation.jl", adbackend)
            time_include("modes/OptimInterface.jl", adbackend)
        end

        # Useful for figuring out why CI is timing out.
        @info "Tests for $(adbackend) took $(time() - start) seconds"
    end
    @testset "variational optimisers" begin
        time_include("variational/optimisers.jl")
    end

    Turing.setadbackend(:forwarddiff)
    @testset "stdlib" begin
        time_include("stdlib/distributions.jl", :forwarddiff)
        time_include("stdlib/RandomMeasures.jl", :forwarddiff)
    end

    @testset "utilities" begin
        # include("utilities/stan-interface.jl")
        time_include("inference/utilities.jl")
    end
end

print(write_running_times(TIMES))
