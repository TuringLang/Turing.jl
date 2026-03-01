iiimodule DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
using DynamicHMC: DynamicHMC
using DynamicPPL: DynamicPPL
using Random: Random
using Turing

@testset "TuringDynamicHMCExt" begin
    rng = Xoshiro(100)
    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(rng, gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
