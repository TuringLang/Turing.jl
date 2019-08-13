using Turing, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@stage_testset "dynamichmc" "dynamichmc.jl" begin
    import DynamicHMC
    chn = sample(gdemo_default, DynamicNUTS(2000));
    check_numerical(chn, [:s, :m], [49/24, 7/6], eps=0.2)
end
