using Turing, Test

include("../test_utils/AllUtils.jl")

@stage_testset "AdvancedHMC" "ahmc.jl" begin
    chn = sample(gdemo_default, ANUTS(10_000, 0.8), 1_000);
    check_numerical(chn, [:s, :m], [49/24, 7/6], eps=0.2)
end
