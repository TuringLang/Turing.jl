using Turing, Random, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "ESS" begin
    @model demo(x) = begin
        m ~ Normal()
        x ~ Normal(m, 0.5)
    end
    demo_default = demo(1.0)

    @turing_testset "ESS constructor" begin
        Random.seed!(0)
        N = 500
        s1 = ESS()
        s2 = ESS(:m)
        s3 = Gibbs(ESS(:m), MH(:s))

        c1 = sample(demo_default, s1, N)
        c2 = sample(demo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
    end

    @numerical_testset "ESS inference" begin
        Random.seed!(1)
        alg = ESS()
        chain = sample(demo_default, alg, 5_000)
        check_numerical(chain, [:m], [0.8], atol = 0.1)

        Random.seed!(100)
        alg = Gibbs(CSMC(15, :s), ESS(:m))
        chain = sample(gdemo_default, alg, 5_000)
        @test_broken check_numerical(chain, [:s, :m], [49/24, 7/6], atol=0.1)

        # MoGtest
        Random.seed!(125)
        gibbs = Gibbs(
            CSMC(15, :z1, :z2, :z3, :z4),
            ESS(:mu1), ESS(:mu2))
        chain = sample(MoGtest_default, gibbs, 6000)
        check_MoGtest_default(chain, atol = 0.1)
    end
end
