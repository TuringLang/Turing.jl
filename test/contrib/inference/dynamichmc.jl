@testset "DynamicHMCExt" begin
    import DynamicHMC
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(externalsampler(DynamicHMC.NUTS()))) == "DynamicNUTS"

    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(gdemo_default, spl, 10_000)
    check_gdemo(chn)
end
