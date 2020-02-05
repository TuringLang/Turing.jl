using Turing, Random, Test, LinearAlgebra
using Turing.Variational: TruncatedADAGrad, DecayedADAGrad

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
        @testset for opt in [TruncatedADAGrad(), DecayedADAGrad()]
            Random.seed!(1)
            N = 500

            alg = ADVI(10, 5000)
            q = vi(gdemo_default, alg; optimizer = opt)
            samples = transpose(rand(q, N))
            chn = Chains(reshape(samples, size(samples)..., 1), ["s", "m"])

            # TODO: uhmm, seems like a large `eps` here...
            check_gdemo(chn, atol = 0.5)
        end
    end
end
