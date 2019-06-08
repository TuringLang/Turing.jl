using Turing, Random, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "mh.jl" begin
    @turing_testset "mh constructor" begin
        Random.seed!(0)
        N = 500
        s1 = MH(N,
            (:s, GKernel(3.0)),
            (:m, GKernel(3.0)))
        s2 = MH(N, :s, :m)
        s3 = MH(N)
        s4 = Gibbs(N, MH(5, :m), MH(5, :s))

        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)
        c3 = sample(gdemo_default, s3)
        c4 = sample(gdemo_default, s4)
    end
    @numerical_testset "mh inference" begin
        Random.seed!(125)
        alg = MH(2000)
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps = 0.1)

        # MH with Gaussian proposal
        alg = MH(5000,
            (:s, GKernel(5)),
            (:m, GKernel(1.0)))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps = 0.2)

        # MH within Gibbs
        alg = Gibbs(1000, MH(5, :m), MH(5, :s))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps = 0.1)

        # MoGtest
        gibbs = Gibbs(1000,
            CSMC(10, 1, :z1, :z2, :z3, :z4),
            MH(10, (:mu1,GKernel(1)), (:mu2,GKernel(1))))
        chain = sample(MoGtest_default, gibbs)
        check_MoGtest_default(chain, eps=0.1)
    end
end
