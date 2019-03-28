using DynamicHMC, Turing, Test
include("../test_utils/models.jl")
@testset "dynamichmc.jl" begin
    @time chn = sample(gdemo_default, DynamicNUTS(500_000));
    # check_numerical(chn, [:s, :m], [49/24, 7/6], eps=0.2)
end
