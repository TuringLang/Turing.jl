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
    # TODO(penelopeysm): sample on both model and LDF for both samplers.
    # Right now it can handle LDF but not model (because RepeatSampler
    # needs to be added to LDFCompatibleSampler)
    for sampler in [MH(), HMC(0.01, 4)]
        ctx = DynamicPPL.SamplingContext(rng, sampler)
        vi = if sampler isa MH
            DynamicPPL.VarInfo(gdemo_default)
        else
            vi = DynamicPPL.VarInfo(gdemo_default)
            vi = DynamicPPL.link(vi, gdemo_default)
            vi
        end
        ldf = LogDensityFunction(gdemo_default, vi, ctx; adtype=Turing.DEFAULT_ADTYPE)

        chn1 = sample(
            copy(rng),
            ldf,
            sampler,
            MCMCThreads(),
            num_samples,
            num_chains;
            thinning=num_repeats,
        )
        repeat_sampler = RepeatSampler(sampler, num_repeats)
        chn2 = sample(
            copy(rng), ldf, repeat_sampler, MCMCThreads(), num_samples, num_chains
        )
        @test chn1.value == chn2.value
    end
end

end
