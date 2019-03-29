using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@testset "pmmh.jl" begin
    @turing_testset "pmmh constructor" begin
        N = 500
        s1 = PMMH(N,
            SMC(10, :s),
            MH(1,(:m, (s) -> Normal(s, sqrt(1)))))
        s2 = PMMH(N, SMC(10, :s), MH(1,:m))
        s3 = PIMH(N, SMC(10))

        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)
        c3 = sample(gdemo_default, s3)
    end
    @numerical_testset "pmmh inference" begin
        alg = PMMH(1000, SMC(20, :m), MH(1,(:s, GKernel(1))))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps=0.1)

        # PMMH with prior as proposal
        alg = PMMH(1000, SMC(20, :m), MH(1,:s))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps=0.1)

        # PIMH
        alg = PIMH(1000, SMC(20))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain)

        # MoGtest
        pmmh = PMMH(500,
            SMC(10, :z1, :z2, :z3, :z4),
            MH(1, :mu1, :mu2))
        chain = sample(MoGtest_default, pmmh)

        check_MoGtest_default(chain, eps = 0.1)
    end
end
