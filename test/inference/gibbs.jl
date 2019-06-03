using Random, Turing, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "gibbs.jl" begin
    @turing_testset "gibbs constructor" begin
        N = 500
        s1 = Gibbs(N, HMC(10, 0.1, 5, :s, :m))
        s2 = Gibbs(N, PG(10, 10, :s, :m))
        s3 = Gibbs(N, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
        s4 = Gibbs(N, PG(10, 3, :s), HMC(2, 0.4, 8, :m); thin=false)
        s5 = Gibbs(N, CSMC(10, 2, :s), HMC(1, 0.4, 8, :m))


        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)
        c3 = sample(gdemo_default, s3)
        c4 = sample(gdemo_default, s4)
        c5 = sample(gdemo_default, s5)

        @test length(c4[:s].value) == N * (3 + 2)

        # Test gid of each samplers
        g = Turing.Sampler(s3, gdemo_default)

        @test g.info[:samplers][1].selector != g.selector
        @test g.info[:samplers][2].selector != g.selector
        @test g.info[:samplers][1].selector != g.info[:samplers][2].selector
    end
    @numerical_testset "gibbs inference" begin
        Random.seed!(100)
        alg = Gibbs(3000,
            CSMC(15, 1, :s),
            HMC(1, 0.2, 4, :m))
        chain = sample(gdemo(1.5, 2.0), alg)
        check_numerical(chain, [:s, :m], [49/24, 7/6], eps=0.1)

        alg = CSMC(15, 5000)
        chain = sample(gdemo(1.5, 2.0), alg)
        check_numerical(chain, [:s, :m], [49/24, 7/6], eps=0.1)

        setadsafe(true)

        Random.seed!(200)
        gibbs = Gibbs(1500,
            PG(10, 1, :z1, :z2, :z3, :z4),
            HMC(3, 0.15, 3, :mu1, :mu2))
        chain = sample(MoGtest_default, gibbs)
        check_MoGtest_default(chain, eps = 0.1)

        setadsafe(false)
    end
end
