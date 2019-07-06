using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@testset "advi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        N = 500

        s1 = ADVI()
        q = vi(gdemo_default, s1)
        c1 = rand(q, N)
    end
    @numerical_testset "advi inference" begin
        Random.seed!(125)
        N = 1000

        alg = ADVI()
        q = vi(gdemo_default, alg)
        samples = reshape(rand(q, N), (N, length(q.μ), 1))
        chn = Chains(samples, ["s", "m"])

        # TODO: uhmm, seems like a large `eps` here...
        check_gdemo(chn, eps = 1.0)
    end
end
