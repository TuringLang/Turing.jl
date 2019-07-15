using Random, Turing, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "gibbs.jl" begin
    @turing_testset "gibbs constructor" begin
        N = 500
        s1 = Gibbs(HMC(0.1, 5, :s, :m))
        s2 = Gibbs(PG(10, :s, :m))
        s3 = Gibbs(PG(3, :s), HMC( 0.4, 8, :m))
        s4 = Gibbs(PG(3, :s), HMC(0.4, 8, :m))
        s5 = Gibbs(CSMC(3, :s), HMC(0.4, 8, :m))


        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
        c5 = sample(gdemo_default, s5, N)

        # Test gid of each samplers
        g = Turing.Sampler(s3, gdemo_default)

        @test g.state.samplers[1].selector != g.selector
        @test g.state.samplers[2].selector != g.selector
        @test g.state.samplers[1].selector != g.state.samplers[2].selector
    end
    @numerical_testset "gibbs inference" begin
        Random.seed!(100)
        alg = Gibbs(
            CSMC(10, :s),
            HMC(0.2, 4, :m))
        chain = sample(gdemo(1.5, 2.0), alg, 3000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], eps=0.1)

        alg = CSMC(10)
        chain = sample(gdemo(1.5, 2.0), alg, 5000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], eps=0.1)

        setadsafe(true)

        Random.seed!(200)
        gibbs = Gibbs(
            PG(10, :z1, :z2, :z3, :z4),
            HMC(0.15, 3, :mu1, :mu2))
        chain = sample(MoGtest_default, gibbs, 1500)
        check_MoGtest_default(chain, eps = 0.1)

        setadsafe(false)
    end
end
