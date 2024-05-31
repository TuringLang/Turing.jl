module DynamicHMCTests

using Test: @test, @testset
using Distributions: sample
import DynamicHMC
import DynamicPPL
using DynamicPPL: Sampler
import Random
using Turing

include(pkgdir(Turing) * "/test/test_utils/models.jl")
include(pkgdir(Turing) * "/test/test_utils/numerical_tests.jl")

@testset "TuringDynamicHMCExt" begin
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(externalsampler(DynamicHMC.NUTS()))) == "DynamicNUTS"

    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

end
