module DynamicHMCTests

using Test: @testset, @test
using Random: Random

using Distributions: sample
using DynamicPPL: DynamicPPL
using DynamicPPL: Sampler
using DynamicHMC: DynamicHMC

using Turing

include(pkgdir(Turing)*"/test/test_utils/models.jl")
include(pkgdir(Turing)*"/test/test_utils/numerical_tests.jl")

@testset "TuringDynamicHMCExt" begin
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(externalsampler(DynamicHMC.NUTS()))) == "DynamicNUTS"

    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
