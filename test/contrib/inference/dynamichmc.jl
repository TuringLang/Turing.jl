@testset "DynamicHMCExt" begin
    using DynamicHMC
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(DynamicNUTS(), gdemo_default)) == "DynamicNUTS"

    spl = externalsampler(DynamicNUTS())
    chn = sample(gdemo_default, spl, 10_000)
    check_gdemo(chn)
end
