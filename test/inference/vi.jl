using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@testset "vi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        N = 500

        s1 = ADVI(gdemo_default)
        c1 = rand(s1, N)
    end
    @numerical_testset "advi inference" begin
        Random.seed!(125)
        N = 1000

        alg = ADVI(gdemo_default)
        samples = reshape(rand(alg, N), (N, length(alg.μ), 1))
        chn = Chains(samples, ["s", "m"])

        # TODO: uhmm, seems like a large `eps` here...
        check_gdemo(chn, eps = 1.0)
    end
end
