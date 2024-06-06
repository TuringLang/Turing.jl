module DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
import DynamicHMC
import DynamicPPL
using DynamicPPL: Sampler
import Random
using Turing

@testset "TuringDynamicHMCExt" begin
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(externalsampler(DynamicHMC.NUTS()))) == "DynamicNUTS"

    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
