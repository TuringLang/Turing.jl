using Turing, Random, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "mh.jl" begin
    @turing_testset "mh constructor" begin
        Random.seed!(0)
        N = 500
        s1 = MH(
            (:s, GKernel(3.0)),
            (:m, GKernel(3.0)))
        s2 = MH(:s, :m)
        s3 = MH()
        s4 = Gibbs(MH(:m), MH(:s))

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
    end
    @numerical_testset "mh inference" begin
        Random.seed!(125)
        alg = MH()
        chain = sample(gdemo_default, alg, 2000)
        check_gdemo(chain, eps = 0.1)

        # MH with Gaussian proposal
        alg = MH(
            (:s, GKernel(5)),
            (:m, GKernel(1.0)))
        chain = sample(gdemo_default, alg, 5000)
        check_gdemo(chain, eps = 0.2)

        # MH within Gibbs
        alg = Gibbs(MH(:m), MH(:s))
        chain = sample(gdemo_default, alg, 2000)
        check_gdemo(chain, eps = 0.1)

        # MoGtest
        gibbs = Gibbs(
            CSMC(15, :z1, :z2, :z3, :z4),
            MH((:mu1,GKernel(1)), (:mu2,GKernel(1))))
        chain = sample(MoGtest_default, gibbs, 6000)
        check_MoGtest_default(chain, eps=0.1)
    end
end
