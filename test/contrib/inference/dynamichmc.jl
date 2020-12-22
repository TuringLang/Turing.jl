using Turing, Test
import Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@stage_testset "dynamichmc" "dynamichmc.jl" begin
    import DynamicHMC
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(DynamicNUTS(), gdemo_default)) == "DynamicNUTS"

    chn = sample(gdemo_default, DynamicNUTS(), 5_000)
    check_gdemo(chn)

    chn2 = sample(gdemo_default, Gibbs(PG(3, :s), DynamicNUTS(:m)), 5_000)
    check_gdemo(chn2)

    chn3 = sample(gdemo_default, Gibbs(DynamicNUTS(:s), ESS(:m)), 5_000)
    check_gdemo(chn3)
end
