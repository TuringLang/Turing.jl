module RepeatSamplerTests

using ..Models: gdemo_default
using MCMCChains: MCMCChains
using Random: Xoshiro
using Test: @test, @testset
using Turing

# RepeatSampler only really makes sense as a component sampler of Gibbs.
# Here we just check that running it by itself is equivalent to thinning.
@testset "RepeatSampler" begin
    num_repeats = 17
    num_samples = 10
    num_chains = 2

    # Use Xoshiro instead of StableRNGs as the output should always be
    # similar regardless of what kind of random seed is used (as long
    # as there is a random seed).
    for sampler in [MH(), HMC(0.01, 4)]
        chn1 = sample(
            Xoshiro(0),
            gdemo_default,
            sampler,
            MCMCThreads(),
            num_samples,
            num_chains;
            thinning=num_repeats,
        )
        repeat_sampler = RepeatSampler(sampler, num_repeats)
        chn2 = sample(
            Xoshiro(0),
            gdemo_default,
            repeat_sampler,
            MCMCThreads(),
            num_samples,
            num_chains,
        )
        # isequal to avoid comparing `missing`s in chain stats
        @test chn1 isa MCMCChains.Chains
        @test chn2 isa MCMCChains.Chains
        @test isequal(chn1.value, chn2.value)
    end
end

end
