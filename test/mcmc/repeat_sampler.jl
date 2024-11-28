module HMCTests

using ..Models: gdemo_default
using ..ADUtils: ADTypeCheckContext
using ..NumericalTests: check_gdemo, check_numerical
import ..ADUtils
using Distributions: Bernoulli, Beta, Categorical, Dirichlet, Normal, Wishart, sample
import DynamicPPL
using DynamicPPL: Sampler
import ForwardDiff
using HypothesisTests: ApproximateTwoSampleKSTest, pvalue
import ReverseDiff
using LinearAlgebra: I, dot, vec
import Random
using StableRNGs: StableRNG
using StatsFuns: logistic
import Mooncake
using Test: @test, @test_logs, @testset, @test_throws
using Turing

# RepeatedSampler only really makes sense as a component sampler of Gibbs.
# Here we just check that running it by itself is equivalent to thinning.
@testset "RepeatedSampler" begin
    num_repeats = 17
    num_samples = 10
    num_chains = 2

    rng = StableRNG(0)
    for sampler in [MH(), Sampler(HMC(0.01, 4))]
        chn1 = sample(
            copy(rng),
            gdemo_default,
            sampler,
            MCMCThreads(),
            num_samples,
            num_chains;
            thinning=num_repeats,
        )
        repeat_sampler = RepeatSampler(sampler, num_repeats)
        chn2 = sample(
            copy(rng), gdemo_default, repeat_sampler, MCMCThreads(), num_samples, num_chains
        )
        @test chn1.value == chn2.value
    end
end

end
