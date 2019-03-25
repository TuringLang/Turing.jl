@testset "mh.jl" begin
    @testset "mh constructor" begin
        Random.seed!(0)
        N = 2000
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

        # Very loose bound, only for testing constructor.
        for c in [c1, c2, c3, c4]
            check_gdemo(c, eps = 1.0)
        end
    end
    @testset "mh inference" begin
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
