module GibbsConditionalTests

using DynamicPPL: DynamicPPL
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
using Turing

@testset "GibbsConditional" begin
    @testset "Gamma model tests" begin
        @model function inverse_gdemo(x)
            precision ~ Gamma(2, inv(3))
            std = sqrt(1 / precision)
            m ~ Normal(0, std)
            for i in 1:length(x)
                x[i] ~ Normal(m, std)
            end
        end

        # Define analytical conditionals
        function cond_precision(c)
            a = 2.0
            b = 3.0
            m = c[@varname(m)]
            x = c[@varname(x)]
            n = length(x)
            a_new = a + (n + 1) / 2
            b_new = b + sum((x[i] - m)^2 for i in 1:n) / 2 + m^2 / 2
            return Gamma(a_new, 1 / b_new)
        end

        function cond_m(c)
            precision = c[@varname(precision)]
            x = c[@varname(x)]
            n = length(x)
            m_mean = sum(x) / (n + 1)
            m_var = 1 / (precision * (n + 1))
            return Normal(m_mean, sqrt(m_var))
        end

        rng = StableRNG(23)
        x_obs = [1.0, 2.0, 3.0, 2.5, 1.5]
        model = inverse_gdemo(x_obs)

        reference_sampler = NUTS()
        reference_chain = sample(rng, model, reference_sampler, 10_000)

        # Use both conditionals, check results against reference sampler.
        sampler = Gibbs(
            :precision => GibbsConditional(cond_precision), :m => GibbsConditional(cond_m)
        )
        chain = sample(rng, model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain, :precision) ≈ mean(reference_chain, :precision) atol = 0.1
        @test mean(chain, :m) ≈ mean(reference_chain, :m) atol = 0.1

        # Mix GibbsConditional with an MCMC sampler
        sampler = Gibbs(:precision => GibbsConditional(cond_precision), :m => MH())
        chain = sample(rng, model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain, :precision) ≈ mean(reference_chain, :precision) atol = 0.1
        @test mean(chain, :m) ≈ mean(reference_chain, :m) atol = 0.1

        sampler = Gibbs(:m => GibbsConditional(cond_m), :precision => HMC(0.1, 10))
        chain = sample(rng, model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain, :precision) ≈ mean(reference_chain, :precision) atol = 0.1
        @test mean(chain, :m) ≈ mean(reference_chain, :m) atol = 0.1

        # Block sample, sampling the same variable with multiple component samplers.
        sampler = Gibbs(
            (:precision, :m) => HMC(0.1, 10),
            :m => GibbsConditional(cond_m),
            :precision => MH(),
            :precision => GibbsConditional(cond_precision),
            :precision => GibbsConditional(cond_precision),
            :precision => HMC(0.1, 10),
            :m => GibbsConditional(cond_m),
            :m => PG(10),
        )
        chain = sample(rng, model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain, :precision) ≈ mean(reference_chain, :precision) atol = 0.1
        @test mean(chain, :m) ≈ mean(reference_chain, :m) atol = 0.1
    end

    @testset "Simple normal model" begin
        @model function simple_normal(dim)
            mean ~ Normal(0, 10)
            var ~ truncated(Normal(1, 1); lower=0.01)
            return x ~ MvNormal(fill(mean, dim), I * var)
        end

        # Conditional posterior for mean given var and x
        function cond_mean(c)
            var = c[@varname(var)]
            x = c[@varname(x)]
            n = length(x)
            # Prior: mean ~ Normal(0, 10)
            # Likelihood: x[i] ~ Normal(mean, σ)
            # Posterior: mean ~ Normal(μ_post, σ_post)
            prior_var = 100.0  # 10^2
            post_var = 1 / (1 / prior_var + n / var)
            post_mean = post_var * (0 / prior_var + sum(x) / var)
            return Normal(post_mean, sqrt(post_var))
        end

        rng = StableRNG(23)
        dim = 1_000
        true_mean = 2.0
        x_obs = randn(rng, dim) .+ true_mean
        model = simple_normal(dim) | (; x=x_obs)
        sampler = Gibbs(:mean => GibbsConditional(cond_mean), :var => MH())
        chain = sample(rng, model, sampler, 1_000)
        # The correct posterior mean isn't true_mean, but it is very close, because we
        # have a lot of data.
        @test mean(chain, :mean) ≈ true_mean atol = 0.05
    end

    @testset "Double simple normal" begin
        # This is the same model as simple_normal above, but just doubled.
        prior_std1 = 10.0
        prior_std2 = 20.0
        @model function double_simple_normal(dim1, dim2)
            mean1 ~ Normal(0, prior_std1)
            var1 ~ truncated(Normal(1, 1); lower=0.01)
            x1 ~ MvNormal(fill(mean1, dim1), I * var1)

            mean2 ~ Normal(0, prior_std2)
            var2 ~ truncated(Normal(1, 1); lower=0.01)
            x2 ~ MvNormal(fill(mean2, dim2), I * var2)
            return nothing
        end

        function cond_mean(var, x, prior_std)
            n = length(x)
            # Prior: mean ~ Normal(0, prior_std)
            # Likelihood: x[i] ~ Normal(mean, σ)
            # Posterior: mean ~ Normal(μ_post, σ_post)
            prior_var = prior_std^2
            post_var = 1 / (1 / prior_var + n / var)
            post_mean = post_var * (0 / prior_var + sum(x) / var)
            return Normal(post_mean, sqrt(post_var))
        end

        rng = StableRNG(23)
        dim1 = 1_000
        true_mean1 = -10.0
        x1_obs = randn(rng, dim1) .+ true_mean1
        dim2 = 2_000
        true_mean2 = -20.0
        x2_obs = randn(rng, dim2) .+ true_mean2
        base_model = double_simple_normal(dim1, dim2)

        # Test different ways of returning values from the conditional function.
        @testset "conditionals return types" begin
            # Test using GibbsConditional for both separately.
            cond_mean1(c) = cond_mean(c[@varname(var1)], c[@varname(x1)], prior_std1)
            cond_mean2(c) = cond_mean(c[@varname(var2)], c[@varname(x2)], prior_std2)
            model = base_model | (; x1=x1_obs, x2=x2_obs)
            sampler = Gibbs(
                :mean1 => GibbsConditional(cond_mean1),
                :mean2 => GibbsConditional(cond_mean2),
                (:var1, :var2) => HMC(0.1, 10),
            )
            chain = sample(StableRNG(23), model, sampler, 1_000)
            # The correct posterior mean isn't true_mean, but it is very close, because we
            # have a lot of data.
            @test mean(chain, :mean1) ≈ true_mean1 atol = 0.1
            @test mean(chain, :mean2) ≈ true_mean2 atol = 0.1

            # Test using GibbsConditional for both in a block, returning a Dict.
            function cond_mean_dict(c)
                return Dict(
                    @varname(mean1) =>
                        cond_mean(c[@varname(var1)], c[@varname(x1)], prior_std1),
                    @varname(mean2) =>
                        cond_mean(c[@varname(var2)], c[@varname(x2)], prior_std2),
                )
            end
            sampler = Gibbs(
                (:mean1, :mean2) => GibbsConditional(cond_mean_dict),
                (:var1, :var2) => HMC(0.1, 10),
            )
            chain = sample(StableRNG(23), model, sampler, 1_000)
            @test mean(chain, :mean1) ≈ true_mean1 atol = 0.1
            @test mean(chain, :mean2) ≈ true_mean2 atol = 0.1

            # As above but with a NamedTuple rather than a Dict.
            function cond_mean_nt(c)
                return (;
                    mean1=cond_mean(c[@varname(var1)], c[@varname(x1)], prior_std1),
                    mean2=cond_mean(c[@varname(var2)], c[@varname(x2)], prior_std2),
                )
            end
            sampler = Gibbs(
                (:mean1, :mean2) => GibbsConditional(cond_mean_nt),
                (:var1, :var2) => HMC(0.1, 10),
            )
            chain = sample(StableRNG(23), model, sampler, 1_000)
            @test mean(chain, :mean1) ≈ true_mean1 atol = 0.1
            @test mean(chain, :mean2) ≈ true_mean2 atol = 0.1
        end

        # Test simultaneously conditioning and fixing variables.
        @testset "condition and fix" begin
            # Note that fixed variables don't contribute to the likelihood, and hence the
            # conditional posterior changes to be just the prior.
            model_condition_fix = condition(fix(base_model; x1=x1_obs); x2=x2_obs)
            function cond_mean1(c)
                @assert @varname(var1) in keys(c)
                @assert @varname(x1) in keys(c)
                return Normal(0.0, prior_std1)
            end
            cond_mean2(c) = cond_mean(c[@varname(var2)], c[@varname(x2)], prior_std2)
            sampler = Gibbs(
                :mean1 => GibbsConditional(cond_mean1),
                :mean2 => GibbsConditional(cond_mean2),
                :var1 => HMC(0.1, 10),
                :var2 => HMC(0.1, 10),
            )
            chain = sample(StableRNG(23), model_condition_fix, sampler, 10_000)
            @test mean(chain, :mean1) ≈ 0.0 atol = 0.1
            @test mean(chain, :mean2) ≈ true_mean2 atol = 0.1

            # As above, but reverse the order of condition and fix.
            model_fix_condition = fix(condition(base_model; x2=x2_obs); x1=x1_obs)
            chain = sample(StableRNG(23), model_condition_fix, sampler, 10_000)
            @test mean(chain, :mean1) ≈ 0.0 atol = 0.1
            @test mean(chain, :mean2) ≈ true_mean2 atol = 0.1
        end
    end

    # Check that GibbsConditional works with VarNames with IndexLenses.
    @testset "Indexed VarNames" begin
        # This example is statistically nonsense, it only tests that the values returned by
        # `conditionals` are passed through correctly.
        @model function f()
            a = Vector{Float64}(undef, 3)
            # These priors will be completely ignored in the sampling.
            a[1] ~ Normal()
            a[2] ~ Normal()
            a[3] ~ Normal()
            return nothing
        end

        m = f()
        function conditionals(c)
            d1 = Normal(0, 1)
            d2 = Normal(c[@varname(a[1])] + 10, 1)
            d3 = Normal(c[@varname(a[2])] + 10, 1)
            return Dict(@varname(a[1]) => d1, @varname(a[2]) => d2, @varname(a[3]) => d3)
        end

        sampler = Gibbs(
            (@varname(a[1]), @varname(a[2]), @varname(a[3])) =>
                GibbsConditional(conditionals),
        )
        chain = sample(StableRNG(23), m, sampler, 1_000)
        @test mean(chain, Symbol("a[1]")) ≈ 0.0 atol = 0.05
        @test mean(chain, Symbol("a[2]")) ≈ 10.0 atol = 0.05
        @test mean(chain, Symbol("a[3]")) ≈ 20.0 atol = 0.05
    end

    @testset "Helpful error outside Gibbs" begin
        @model f() = x ~ Normal()
        m = f()
        cond_x(_) = Normal()
        sampler = GibbsConditional(cond_x)
        @test_throws(
            "Are you trying to use GibbsConditional outside of Gibbs?",
            sample(m, sampler, 3),
        )
    end
end

end
