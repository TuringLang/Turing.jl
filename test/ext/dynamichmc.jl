module DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
using DynamicHMC: DynamicHMC
using DynamicPPL: DynamicPPL
using DynamicPPL: Sampler
using Random: Random
using StableRNGs: StableRNG
using Turing

@testset "TuringDynamicHMCExt" begin
    @test DynamicPPL.alg_str(Sampler(externalsampler(DynamicHMC.NUTS()))) == "DynamicNUTS"

    rng = StableRNG(468)
    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(rng, gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
