using Turing, Test
import Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@stage_testset "dynamichmc" "dynamichmc.jl" begin
    import DynamicHMC
    Random.seed!(100)

    @test DynamicPPL.alg_str(Sampler(DynamicNUTS(), gdemo_default)) == "HMC"

    chn = sample(gdemo_default, DynamicNUTS(), 5000)
    check_numerical(chn, [:s, :m], [49/24, 7/6], atol=0.2)
end
