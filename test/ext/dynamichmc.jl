module DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
using DynamicHMC: DynamicHMC
using DynamicPPL: DynamicPPL
using Random: Random
using StableRNGs: StableRNG
using Turing

@testset "TuringDynamicHMCExt" begin
    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(StableRNG(100), gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
