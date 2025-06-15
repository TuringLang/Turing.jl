module RepeatSamplerTests

using ..Models: gdemo_default
using DynamicPPL: DynamicPPL
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
    for sampler in [MH(), DynamicPPL.Sampler(HMC(0.01, 4))]
        model_or_ldf = if sampler isa MH
            gdemo_default
        else
            vi = DynamicPPL.VarInfo(gdemo_default)
            vi = DynamicPPL.link(vi, gdemo_default)
            LogDensityFunction(gdemo_default, vi; adtype=Turing.DEFAULT_ADTYPE)
        end

        chn1 = sample(
            copy(rng),
            model_or_ldf,
            sampler,
            MCMCThreads(),
            num_samples,
            num_chains;
            thinning=num_repeats,
        )
        repeat_sampler = RepeatSampler(sampler, num_repeats)
        chn2 = sample(
            copy(rng), model_or_ldf, repeat_sampler, MCMCThreads(), num_samples, num_chains
        )
        @test chn1.value == chn2.value
    end
end

end
