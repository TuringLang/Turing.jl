module ESSTests

using ..Models: MoGtest, MoGtest_default, gdemo, gdemo_default
using ..NumericalTests: check_MoGtest_default, check_numerical
using Distributions: Normal, sample
using DynamicPPL: DynamicPPL
using DynamicPPL: Sampler
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @testset, @test_broken
using Turing

@testset "ESS" begin
    @info "Starting ESS tests"

    @testset "InferenceAlgorithm interface" begin
        alg = ESS()
        @test Turing.Inference.get_adtype(alg) === nothing
        @test !Turing.Inference.requires_unconstrained_space(alg)
        kwargs = (; _foo="bar")
        @test Turing.Inference.update_sample_kwargs(alg, 1000, kwargs) == kwargs
    end

    @testset "sample() interface" begin
        @model function demo_normal(x)
            a ~ Normal()
            return x ~ Normal(a)
        end
        model = demo_normal(2.0)
        ldf = LogDensityFunction(model)
        sampling_objects = Dict("DynamicPPL.Model" => model, "LogDensityFunction" => ldf)
        seed = 468

        @testset "sampling with $name" for (name, model_or_ldf) in sampling_objects
            spl = ESS()
            # check sampling works without rng
            @test sample(model_or_ldf, spl, 5) isa Chains
            # check reproducibility with rng
            chn1 = sample(Random.Xoshiro(seed), model_or_ldf, spl, 5)
            chn2 = sample(Random.Xoshiro(seed), model_or_ldf, spl, 5)
            @test mean(chn1[:a]) == mean(chn2[:a])
        end

        @testset "check that initial_params are respected" begin
            a0 = 5.0
            chn = sample(model, ESS(), 5; initial_params=[a0])
            @test chn[:a][1] == a0
        end
    end

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

        # TODO(penelopeysm): fix
        @testset "gdemo with CSMC + ESS" begin
            @test_broken false
            # alg = Gibbs(:s => CSMC(15), :m => ESS())
            # chain = sample(StableRNG(seed), gdemo(1.5, 2.0), alg, 2000)
            # check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)
        end

        @testset "MoGtest_default with CSMC + ESS" begin
            @test_broken false
            # alg = Gibbs(
            #     (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
            #     @varname(mu1) => ESS(),
            #     @varname(mu2) => ESS(),
            # )
            # chain = sample(StableRNG(seed), MoGtest_default, alg, 2000)
            # check_MoGtest_default(chain; atol=0.1)
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
                ESS(),
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

        # TODO(penelopeysm) Fix this
        @test_broken false
        # num_samples = 10_000
        # spl_x = Gibbs(@varname(z) => NUTS(), @varname(x) => ESS())
        # spl_xy = Gibbs(@varname(z) => NUTS(), (@varname(x), @varname(y)) => ESS())
        #
        # @test sample(StableRNG(23), xy(), spl_xy, num_samples).value ≈
        #     sample(StableRNG(23), x12(), spl_x, num_samples).value
    end
end

end
