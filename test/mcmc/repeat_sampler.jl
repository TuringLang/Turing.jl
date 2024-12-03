module RepeatSamplerTests

using ..Models: gdemo_default
using DynamicPPL: Sampler
using StableRNGs: StableRNG
using Test: @test, @testset
using Turing

# RepeatSampler only really makes sense as a component sampler of Gibbs.
# Here we just check that running it by itself is equivalent to thinning.
@testset "RepeatSampler" begin
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
