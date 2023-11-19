using AbstractMCMC
using AdvancedMH
using AdvancedPS
using Clustering
using Distributions
using Distributions.FillArrays
using DistributionsAD
using FiniteDifferences
using ForwardDiff
using MCMCChains
using NamedArrays
using Optim
using Optimization
using OptimizationOptimJL
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
using TimerOutputs: TimerOutputs, @timeit
using Turing: BinomialLogit, ForwardDiffAD, Sampler, SampleFromPrior, NUTS, TrackerAD,
                Variational, ZygoteAD, getspace
using Turing.Essential: TuringDenseMvNormal, TuringDiagMvNormal
using Turing.Variational: TruncatedADAGrad, DecayedADAGrad, AdvancedVI

import LogDensityProblems
import LogDensityProblemsAD

setprogress!(false)

include(pkgdir(Turing)*"/test/test_utils/AllUtils.jl")

# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
macro timeit_include(path::AbstractString) :(@timeit TIMEROUTPUT $path include($path)) end

@testset "Turing" begin
    @testset "essential" begin
        @timeit_include("essential/ad.jl")
    end

    @testset "samplers (without AD)" begin
        @timeit_include("mcmc/particle_mcmc.jl")
        @timeit_include("mcmc/emcee.jl")
        @timeit_include("mcmc/ess.jl")
        @timeit_include("mcmc/is.jl")
    end

    Turing.setrdcache(false)
    for adbackend in (:forwarddiff, :reversediff)
        @timeit TIMEROUTPUT "inference: $adbackend" begin
            Turing.setadbackend(adbackend)
            @info "Testing $(adbackend)"
            @testset "inference: $adbackend" begin
                @testset "samplers" begin
                    @timeit_include("mcmc/gibbs.jl")
                    @timeit_include("mcmc/gibbs_conditional.jl")
                    @timeit_include("mcmc/gibbs_new.jl")
                    @timeit_include("mcmc/hmc.jl")
                    @timeit_include("mcmc/Inference.jl")
                    @timeit_include("mcmc/sghmc.jl")
                    @timeit_include("mcmc/abstractmcmc.jl")
                    @timeit_include("mcmc/mh.jl")
                    @timeit_include("ext/dynamichmc.jl")
                end
            end

            @testset "variational algorithms : $adbackend" begin
                @timeit_include("variational/advi.jl")
            end

            @testset "mode estimation : $adbackend" begin
                @timeit_include("optimisation/OptimInterface.jl")
                @timeit_include("ext/Optimisation.jl")
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
        @timeit_include("mcmc/utilities.jl")
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)
