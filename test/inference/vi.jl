using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@testset "vi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        N = 500
        s1 = Turing.Inference.ADVI(N)

        c1 = sample(gdemo_default, s1)
    end
    @numerical_testset "advi inference" begin
        Random.seed!(125)
        alg = Turing.Inference.ADVI(2000)
        chain = sample(gdemo_default, alg)
        # TODO: do the numerical check after using the 
        # check_gdemo(chain, eps = 0.1)
    end
end
