module GibbsConditionalTests

using AbstractPPL: AbstractPPL
using DynamicPPL: DynamicPPL
using LinearAlgebra: LinearAlgebra
using Random: Random, Xoshiro
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
using Turing

@testset "GibbsConditional" begin
    @testset "a ranged site is one variable, not several" begin
        # `theta[:] ~ MvNormal(...)` is one tilde statement stored under `theta[1], theta[2]`,
        # so counting keys refused the single-distribution form it is meant to support. The
        # conditional is stored at `theta` and each tilde key resolves to it by subsumption.
        @model function ranged()
            theta = Vector{Float64}(undef, 2)
            theta[:] ~ MvNormal(zeros(2), LinearAlgebra.I)
            return 1.0 ~ Normal(sum(theta), 1)
        end
        spl = Gibbs(
            @varname(theta) => GibbsConditional(_ -> MvNormal(zeros(2), LinearAlgebra.I))
        )
        @test sample(Xoshiro(1), ranged(), spl, 20; check_model=false, progress=false) isa
            Any
        # Two genuine variables under one distribution are still refused.
        @model function two()
            a ~ Normal()
            b ~ Normal()
            return 1.0 ~ Normal(a + b, 1)
        end
        @test_throws "multiple variables" sample(
            Xoshiro(1),
            two(),
            Gibbs((@varname(a), @varname(b)) => GibbsConditional(_ -> Normal())),
            5;
            check_model=false,
            progress=false,
        )
    end

    @testset "a keyword model argument reaches the conditional" begin
        # `model.args` holds only positional arguments; a keyword one lands in
        # `model.defaults` and was invisible here.
        @model function kw(; y=2.0)
            m ~ Normal()
            return y ~ Normal(m, 1)
        end
        seen = Ref{Any}(nothing)
        cond_m(vnt) = (seen[] = vnt; Normal(0, 1))
        sample(
            Xoshiro(1),
            kw(),
            Gibbs(@varname(m) => GibbsConditional(cond_m)),
            3;
            check_model=false,
            progress=false,
        )
        @test DynamicPPL.getvalue(seen[], @varname(y)) == 2.0
    end

    @testset "observed and latent elements of one array" begin
        @model function partly()
            mu ~ Normal()
            x = Vector{Float64}(undef, 2)
            x[1] ~ Normal(mu, 1)
            return x[2] ~ Normal(mu, 1)
        end

        # `x[1]` is conditioned data, `x[2]` is latent and conditioned by Gibbs on the other
        # component's draw. The conditional has to see both: a key-level merge keeps one and
        # drops the other. Conditioned rather than a `missing` argument, which Gibbs refuses.
        seen = Ref{Any}(nothing)
        function cond_mu(vnt)
            seen[] = vnt
            return Normal(0, 1)
        end
        spl = Gibbs(@varname(mu) => GibbsConditional(cond_mu), @varname(x[2]) => MH())
        model = DynamicPPL.condition(partly(), Dict(@varname(x[1]) => 1.5))
        sample(Xoshiro(1), model, spl, 3; check_model=false, progress=false)
        @test DynamicPPL.getvalue(seen[], @varname(x[1])) == 1.5
        @test DynamicPPL.hasvalue(seen[], @varname(x[2]))
    end

    @testset "Gamma model tests" begin
        @model function inverse_gdemo(x)
            precision ~ Gamma(2, inv(3))
            std = sqrt(1 / precision)
            m ~ Normal(0, std)
            for i in 1:length(x)
                x[i] ~ Normal(m, std)
            end
        end

        # Define analytical conditionals. See
        # https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
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

        x_obs = [1.0, 2.0, 3.0, 2.5, 1.5]
        model = inverse_gdemo(x_obs)

        reference_sampler = NUTS()
        reference_chain = sample(StableRNG(23), model, reference_sampler, 10_000)

        # Use both conditionals, check results against reference sampler.
        sampler = Gibbs(
            :precision => GibbsConditional(cond_precision), :m => GibbsConditional(cond_m)
        )
        chain = sample(StableRNG(23), model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain[@varname(precision)]) ≈ mean(reference_chain[@varname(precision)]) atol =
            0.1
        @test mean(chain[@varname(m)]) ≈ mean(reference_chain[@varname(m)]) atol = 0.1

        # Mix GibbsConditional with an MCMC sampler
        sampler = Gibbs(:precision => GibbsConditional(cond_precision), :m => MH())
        chain = sample(StableRNG(23), model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain[@varname(precision)]) ≈ mean(reference_chain[@varname(precision)]) atol =
            0.1
        @test mean(chain[@varname(m)]) ≈ mean(reference_chain[@varname(m)]) atol = 0.1

        sampler = Gibbs(:m => GibbsConditional(cond_m), :precision => HMC(0.1, 10))
        chain = sample(StableRNG(23), model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain[@varname(precision)]) ≈ mean(reference_chain[@varname(precision)]) atol =
            0.1
        @test mean(chain[@varname(m)]) ≈ mean(reference_chain[@varname(m)]) atol = 0.1

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
        chain = sample(StableRNG(23), model, sampler, 1_000)
        @test size(chain, 1) == 1_000
        @test mean(chain[@varname(precision)]) ≈ mean(reference_chain[@varname(precision)]) atol =
            0.1
        @test mean(chain[@varname(m)]) ≈ mean(reference_chain[@varname(m)]) atol = 0.1
    end

    @testset "Simple normal model" begin
        @model function simple_normal(dim)
            mean ~ Normal(0, 10)
            var ~ truncated(Normal(1, 1); lower=0.01)
            return x ~ MvNormal(fill(mean, dim), I * var)
        end

        # Conditional posterior for mean given var and x. See
        # https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
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

        dim = 1_000
        true_mean = 2.0
        x_obs = randn(StableRNG(23), dim) .+ true_mean
        model = simple_normal(dim) | (; x=x_obs)
        sampler = Gibbs(:mean => GibbsConditional(cond_mean), :var => MH())
        chain = sample(StableRNG(23), model, sampler, 1_000)
        # The correct posterior mean isn't true_mean, but it is very close, because we
        # have a lot of data.
        @test mean(chain[@varname(mean)]) ≈ true_mean atol = 0.05
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

        dim1 = 1_000
        true_mean1 = -10.0
        x1_obs = randn(StableRNG(23), dim1) .+ true_mean1
        dim2 = 2_000
        true_mean2 = -20.0
        x2_obs = randn(StableRNG(24), dim2) .+ true_mean2
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
            @test mean(chain[@varname(mean1)]) ≈ true_mean1 atol = 0.1
            @test mean(chain[@varname(mean2)]) ≈ true_mean2 atol = 0.1

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
            @test mean(chain[@varname(mean1)]) ≈ true_mean1 atol = 0.1
            @test mean(chain[@varname(mean2)]) ≈ true_mean2 atol = 0.1

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
            @test mean(chain[@varname(mean1)]) ≈ true_mean1 atol = 0.1
            @test mean(chain[@varname(mean2)]) ≈ true_mean2 atol = 0.1
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
            @test mean(chain[@varname(mean1)]) ≈ 0.0 atol = 0.1
            @test mean(chain[@varname(mean2)]) ≈ true_mean2 atol = 0.1

            # As above, but reverse the order of condition and fix.
            model_fix_condition = fix(condition(base_model; x2=x2_obs); x1=x1_obs)
            chain = sample(StableRNG(23), model_fix_condition, sampler, 10_000)
            @test mean(chain[@varname(mean1)]) ≈ 0.0 atol = 0.1
            @test mean(chain[@varname(mean2)]) ≈ true_mean2 atol = 0.1
        end
    end

    # Check that GibbsConditional works with VarNames with IndexLenses.
    @testset "Indexed VarNames" begin
        # This example is statistically nonsense, it only tests that the values returned by
        # `conditionals` are passed through correctly.
        @model function f()
            a = Vector{Float64}(undef, 3)
            a[1] ~ Normal(0.0)
            a[2] ~ Normal(10.0)
            a[3] ~ Normal(20.0)
            b = Vector{Float64}(undef, 3)
            # These priors will be completely ignored in the sampling.
            b[1] ~ Normal()
            b[2] ~ Normal()
            b[3] ~ Normal()
            return nothing
        end

        m = f()
        function conditionals_b(c)
            d1 = Normal(c[@varname(a[1])], 1)
            d2 = Normal(c[@varname(a[2])], 1)
            d3 = Normal(c[@varname(a[3])], 1)
            return @vnt begin
                @template b = zeros(3)
                b[1] := d1
                b[2] := d2
                b[3] := d3
            end
        end

        sampler = Gibbs(
            (@varname(b[1]), @varname(b[2]), @varname(b[3])) =>
                GibbsConditional(conditionals_b),
            (@varname(a[1]), @varname(a[2]), @varname(a[3])) => ESS(),
        )
        chain = sample(StableRNG(23), m, sampler, 10_000)
        @test mean(chain[@varname(b[1])]) ≈ 0.0 atol = 0.05
        @test mean(chain[@varname(b[2])]) ≈ 10.0 atol = 0.05
        @test mean(chain[@varname(b[3])]) ≈ 20.0 atol = 0.05

        condvals = @vnt begin
            @template a = zeros(3)
            a[1] := 100.0
        end
        fixvals = @vnt begin
            @template a = zeros(3)
            a[2] := 200.0
        end
        m_condfix = fix(condition(m, condvals), fixvals)
        sampler = Gibbs(
            (@varname(b[1]), @varname(b[2]), @varname(b[3])) =>
                GibbsConditional(conditionals_b),
            @varname(a[3]) => ESS(),
        )
        chain = sample(StableRNG(23), m_condfix, sampler, 10_000)
        @test mean(chain[@varname(b[1])]) ≈ 100.0 atol = 0.05
        @test mean(chain[@varname(b[2])]) ≈ 200.0 atol = 0.05
        @test mean(chain[@varname(b[3])]) ≈ 20.0 atol = 0.05
    end

    @testset "block reshaped by another component" begin
        # `b` decides how many elements of `theta` the model reaches, so the
        # `GibbsConditional` block arrives at a new dimension between its own steps. The
        # conditional distributions describe the block as it is now; the state's values
        # describe it as it was, so they cannot be used to shape the result.
        @model function dyn()
            b ~ Bernoulli(0.5)
            n = b ? 2 : 1
            theta = Vector{Float64}(undef, n)
            for i in 1:n
                theta[i] ~ Normal(0, 1)
            end
            return 1.0 ~ Normal(sum(theta), 1.0)
        end
        function cond(vnt)
            n = vnt[@varname(b)] ? 2 : 1
            return Dict(@varname(theta[i]) => Normal(0, 1) for i in 1:n)
        end
        sampler = Gibbs(
            (@varname(b), @varname(theta)) => PG(20),
            @varname(theta) => GibbsConditional(cond),
        )
        @test sample(StableRNG(468), dyn(), sampler, 20) isa Any
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
