module ESSTests

using ..Models: MoGtest, MoGtest_default, gdemo, gdemo_default
using ..NumericalTests: check_MoGtest_default, check_numerical
using Distributions: Normal, sample
using DynamicPPL: DynamicPPL
using DynamicPPL: Sampler
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @testset
using Turing

@testset "ESS" begin
    @info "Starting ESS tests"

    @model function demo(x)
        m ~ Normal()
        return x ~ Normal(m, 0.5)
    end
    demo_default = demo(1.0)

    @model function demodot(x)
        m = Vector{Float64}(undef, 2)
        @. m ~ Normal()
        return x ~ Normal(m[2], 0.5)
    end
    demodot_default = demodot(1.0)

    @testset "ESS constructor" begin
        N = 10

        s1 = ESS()
        @test DynamicPPL.alg_str(Sampler(s1)) == "ESS"

        c1 = sample(demo_default, s1, N)
        c2 = sample(demodot_default, s1, N)

        s2 = Gibbs(:m => ESS(), :s => MH())
        c3 = sample(gdemo_default, s2, N)
    end

    @testset "ESS inference" begin
        @info "Starting ESS inference tests"
        seed = 23

        @testset "demo_default" begin
            chain = sample(StableRNG(seed), demo_default, ESS(), 5_000)
            check_numerical(chain, [:m], [0.8]; atol=0.1)
        end

        @testset "demodot_default" begin
            chain = sample(StableRNG(seed), demodot_default, ESS(), 5_000)
            check_numerical(chain, ["m[1]", "m[2]"], [0.0, 0.8]; atol=0.1)
        end

        @testset "gdemo with CSMC + ESS" begin
            alg = Gibbs(:s => CSMC(15), :m => ESS())
            chain = sample(StableRNG(seed), gdemo(1.5, 2.0), alg, 2000)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)
        end

        @testset "MoGtest_default with CSMC + ESS" begin
            alg = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
                @varname(mu1) => ESS(),
                @varname(mu2) => ESS(),
            )
            chain = sample(StableRNG(seed), MoGtest_default, alg, 2000)
            check_MoGtest_default(chain; atol=0.1)
        end

        @testset "TestModels" begin
            # Different "equivalent" models.
            # NOTE: Because `ESS` only supports "single" variables with
            # Gaussian priors, we restrict ourselves to this subspace by conditioning
            # on the non-Gaussian variables in `DEMO_MODELS`.
            models_conditioned = map(DynamicPPL.TestUtils.DEMO_MODELS) do model
                # Condition on the non-Gaussian random variables.
                model | (s=DynamicPPL.TestUtils.posterior_mean(model).s,)
            end

            DynamicPPL.TestUtils.test_sampler(
                models_conditioned,
                DynamicPPL.Sampler(ESS()),
                2000;
                # Filter out the varnames we've conditioned on.
                varnames_filter=vn -> DynamicPPL.getsym(vn) != :s,
            )
        end
    end

    # Test that ESS can sample multiple variables regardless of whether they are under the
    # same symbol or not.
    @testset "Multiple variables" begin
        @model function xy()
            z ~ Beta(2.0, 2.0)
            x ~ Normal(z, 2.0)
            return y ~ Normal(-3.0, 3.0)
        end

        @model function x12()
            z ~ Beta(2.0, 2.0)
            x = Vector{Float64}(undef, 2)
            x[1] ~ Normal(z, 2.0)
            return x[2] ~ Normal(-3.0, 3.0)
        end

        num_samples = 10_000
        spl_x = Gibbs(@varname(z) => NUTS(), @varname(x) => ESS())
        spl_xy = Gibbs(@varname(z) => NUTS(), (@varname(x), @varname(y)) => ESS())

        @test sample(StableRNG(23), xy(), spl_xy, num_samples).value ≈
            sample(StableRNG(23), x12(), spl_x, num_samples).value
    end
end

end
