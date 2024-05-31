module ESSTests

using Distributions: Normal, sample
import DynamicPPL
using DynamicPPL: Sampler
import Random
using Test: @test, @testset
using Turing

include(pkgdir(Turing) * "/test/test_utils/models.jl")
include(pkgdir(Turing) * "/test/test_utils/numerical_tests.jl")

@testset "ESS" begin
    @model function demo(x)
        m ~ Normal()
        x ~ Normal(m, 0.5)
    end
    demo_default = demo(1.0)

    @model function demodot(x)
        m = Vector{Float64}(undef, 2)
        @. m ~ Normal()
        x ~ Normal(m[2], 0.5)
    end
    demodot_default = demodot(1.0)

    @testset "ESS constructor" begin
        Random.seed!(0)
        N = 500

        s1 = ESS()
        s2 = ESS(:m)
        for s in (s1, s2)
            @test DynamicPPL.alg_str(Sampler(s, demo_default)) == "ESS"
        end

        c1 = sample(demo_default, s1, N)
        c2 = sample(demo_default, s2, N)
        c3 = sample(demodot_default, s1, N)
        c4 = sample(demodot_default, s2, N)

        s3 = Gibbs(ESS(:m), MH(:s))
        c5 = sample(gdemo_default, s3, N)
    end

    @testset "ESS inference" begin
        Random.seed!(1)
        chain = sample(demo_default, ESS(), 5_000)
        check_numerical(chain, [:m], [0.8], atol = 0.1)

        Random.seed!(1)
        chain = sample(demodot_default, ESS(), 5_000)
        check_numerical(chain, ["m[1]", "m[2]"], [0.0, 0.8], atol = 0.1)

        Random.seed!(100)
        alg = Gibbs(
            CSMC(15, :s),
            ESS(:m))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol = 0.1)

        # MoGtest
        Random.seed!(125)
        alg = Gibbs(
            CSMC(15, :z1, :z2, :z3, :z4),
            ESS(:mu1), ESS(:mu2))
        chain = sample(MoGtest_default, alg, 6000)
        check_MoGtest_default(chain, atol = 0.1)

        # Different "equivalent" models.
        # NOTE: Because `ESS` only supports "single" variables with
        # Gaussian priors, we restrict ourselves to this subspace by conditioning
        # on the non-Gaussian variables in `DEMO_MODELS`.
        models_conditioned = map(DynamicPPL.TestUtils.DEMO_MODELS) do model
            # Condition on the non-Gaussian random variables.
            model | (s = DynamicPPL.TestUtils.posterior_mean(model).s,)
        end

        DynamicPPL.TestUtils.test_sampler(
            models_conditioned, DynamicPPL.Sampler(ESS()), 10_000;
            # Filter out the varnames we've conditioned on.
            varnames_filter=vn -> DynamicPPL.getsym(vn) != :s
        )
    end
end

end
