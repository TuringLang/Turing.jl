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
using StableRNGs

using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using AdvancedVI: TruncatedADAGrad, DecayedADAGrad, apply!
using Distributions: Binomial, logpdf
using DynamicPPL: getval, getlogp
using ForwardDiff: Dual
using MCMCChains: Chains
using StatsFuns: binomlogpdf, logistic, logsumexp
using TimerOutputs: TimerOutput, @timeit, print_timer
using Turing: BinomialLogit, ForwardDiffAD, Sampler, SampleFromPrior, NUTS, TrackerAD,
                Variational, ZygoteAD, getspace, gradient_logp
using Turing.Essential: TuringDenseMvNormal, TuringDiagMvNormal
using Turing.Variational: TruncatedADAGrad, DecayedADAGrad, AdvancedVI

setprogress!(false)

include(pkgdir(Turing)*"/test/test_utils/AllUtils.jl")

# Collect timing and allocations information to show in a clear way.
const to = TimerOutput()

@testset "Turing" begin
    @testset "essential" begin
        @timeit to "essential/ad" include("essential/ad.jl")
    end

    @testset "samplers (without AD)" begin
        @timeit to "inference/AdvancedSMC" include("inference/AdvancedSMC.jl")
        @timeit to "inference/emcee" include("inference/emcee.jl")
        @timeit to "inference/ess" include("inference/ess.jl")
        @timeit to "inference/is" include("inference/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :tracker, :reversediff)
        Turing.setadbackend(adbackend)
        @info "Testing $(adbackend)"
        start = time()
        @timeit to "inference: $adbackend" begin
            @testset "inference: $adbackend" begin
                @testset "samplers" begin
                    @timeit to "gibbs" include("inference/gibbs.jl")
                    @timeit to "gibbs_conditional" include("inference/gibbs_conditional.jl")
                    @timeit to "hmc" include("inference/hmc.jl")
                    @timeit to "Inference" include("inference/Inference.jl")
                    @timeit to "dynamichmc" include("contrib/inference/dynamichmc.jl")
                    @timeit to "sghmc" include("contrib/inference/sghmc.jl")
                    @timeit to "mh" include("inference/mh.jl")
                end
            end
        end

        @testset "variational algorithms : $adbackend" begin
            @timeit to "variational/advi" include("variational/advi.jl")
        end

        @testset "modes" begin
            @timeit to "ModeEstimation" include("modes/ModeEstimation.jl")
            @timeit to "OptimInterface" include("modes/OptimInterface.jl")
        end

        # Useful for figuring out why CI is timing out.
        @info "Tests for $(adbackend) took $(time() - start) seconds"
    end
    @testset "variational optimisers" begin
        @timeit to "optimisers" include("variational/optimisers.jl")
    end

    Turing.setadbackend(:forwarddiff)
    @testset "stdlib" begin
        @timeit to "distributions" include("stdlib/distributions.jl")
        @timeit to "RandomMeasures" include("stdlib/RandomMeasures.jl")
    end

    @testset "utilities" begin
        # include("utilities/stan-interface.jl")
        @timeit to "utilities" include("inference/utilities.jl")
    end
end

# Hiding `avg` column via `compact=true` because we do only one run per entry.
print_timer(to; compact=true, sortby=:firstexec)
