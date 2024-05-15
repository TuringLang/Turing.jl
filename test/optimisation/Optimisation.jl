@testset "ext/Optimisation.jl" begin
    @numerical_testset "gdemo" begin
        @testset "MLE" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))

            function check_success(result)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MLE())
            m2 = estimate_mode(gdemo_default, MLE(), true_value, LBFGS())
            m3 = estimate_mode(gdemo_default, MLE(), Newton())
            # TODO(mhauru) How can we check that the adtype is actually AutoReverseDiff?
            m4 = estimate_mode(gdemo_default, MLE(), BFGS(); adtype=AutoReverseDiff())
            m5 = estimate_mode(gdemo_default, MLE(), true_value, NelderMead())
            m6 = maximum_likelihood(gdemo_default, NelderMead())

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test m2.optim_result.stats.iterations <= 1
            @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations

            @test m2.optim_result.stats.gevals > 0
            @test m3.optim_result.stats.gevals > 0
            @test m4.optim_result.stats.gevals > 0
            @test m5.optim_result.stats.gevals == 0
            @test m6.optim_result.stats.gevals == 0
        end

        @testset "MAP" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))

            function check_success(result)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MAP())
            m2 = estimate_mode(gdemo_default, MAP(), true_value, LBFGS())
            m3 = estimate_mode(gdemo_default, MAP(), Newton())
            m4 = estimate_mode(gdemo_default, MAP(), BFGS(); adtype=AutoReverseDiff())
            m5 = estimate_mode(gdemo_default, MAP(), true_value, NelderMead())
            m6 = maximum_a_posteriori(gdemo_default, NelderMead())

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test m2.optim_result.stats.iterations <= 1
            @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations

            @test m2.optim_result.stats.gevals > 0
            @test m3.optim_result.stats.gevals > 0
            @test m4.optim_result.stats.gevals > 0
            @test m5.optim_result.stats.gevals == 0
            @test m6.optim_result.stats.gevals == 0
        end

        @testset "MLE with box constraints" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))

            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            function check_success(result, check_retcode=true)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                if check_retcode
                    @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                end
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MLE(); lb=lb, ub=ub)
            m2 = estimate_mode(
                gdemo_default, MLE(), true_value, Fminbox(LBFGS()); lb=lb, ub=ub
            )
            m3 = estimate_mode(
                gdemo_default, MLE(), BBO_separable_nes();
                maxiters=100_000, abstol=1e-5, lb=lb, ub=ub
            )
            m4 = estimate_mode(
                gdemo_default, MLE(), Fminbox(BFGS()); adtype=AutoReverseDiff(), lb=lb, ub=ub
            )
            m5 = estimate_mode(gdemo_default, MLE(), true_value, IPNewton(); lb=lb, ub=ub)
            m6 = maximum_likelihood(gdemo_default; lb=lb, ub=ub)

            check_success(m1)
            check_success(m2)
            # BBO retcodes are misconfigured, so skip checking the retcode in this case.
            # See https://github.com/SciML/Optimization.jl/issues/745
            check_success(m3, false)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test m2.optim_result.stats.iterations <= 1
            @test m5.optim_result.stats.iterations <= 1

            @test m2.optim_result.stats.gevals > 0
            @test m3.optim_result.stats.gevals == 0
            @test m4.optim_result.stats.gevals > 0
            @test m5.optim_result.stats.gevals > 0
        end

        @testset "MAP with box constraints" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))

            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            function check_success(result, check_retcode=true)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                if check_retcode
                    @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                end
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MAP(); lb=lb, ub=ub)
            m2 = estimate_mode(
                gdemo_default, MAP(), true_value, Fminbox(LBFGS()); lb=lb, ub=ub
            )
            m3 = estimate_mode(
                gdemo_default, MAP(), BBO_separable_nes();
                maxiters=100_000, abstol=1e-5, lb=lb, ub=ub
            )
            m4 = estimate_mode(
                gdemo_default, MAP(), Fminbox(BFGS()); adtype=AutoReverseDiff(), lb=lb, ub=ub
            )
            m5 = estimate_mode(gdemo_default, MAP(), true_value, IPNewton(); lb=lb, ub=ub)
            m6 = maximum_a_posteriori(gdemo_default; lb=lb, ub=ub)

            check_success(m1)
            check_success(m2)
            # BBO retcodes are misconfigured, so skip checking the retcode in this case.
            # See https://github.com/SciML/Optimization.jl/issues/745
            check_success(m3, false)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test m2.optim_result.stats.iterations <= 1
            @test m5.optim_result.stats.iterations <= 1

            @test m2.optim_result.stats.gevals > 0
            @test m3.optim_result.stats.gevals == 0
            @test m4.optim_result.stats.gevals > 0
            @test m5.optim_result.stats.gevals > 0
        end

        @testset "MLE with generic constraints" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))

            # Set two constraints: The first parameter must be non-negative, and the L2 norm
            # of the parameters must be between 0.5 and 2.
            cons(res, x, _) = (res .= [x[1], sqrt(sum(x .^ 2))])
            lcons = [0, 0.5]
            ucons = [Inf, 2.0]
            cons_args = (cons=cons, lcons=lcons, ucons=ucons)
            init_value = [0.5, -1.0]

            function check_success(result)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MLE(), init_value; cons_args...)
            m2 = estimate_mode(gdemo_default, MLE(), true_value; cons_args...)
            m3 = estimate_mode(gdemo_default, MLE(), init_value, IPNewton(); cons_args...)
            m4 = estimate_mode(
                gdemo_default, MLE(), init_value, IPNewton();
                adtype=AutoReverseDiff(), cons_args...
            )
            m5 = maximum_likelihood(gdemo_default, init_value; cons_args...)

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)

            @test m2.optim_result.stats.iterations <= 1

            @test m3.optim_result.stats.gevals > 0
            @test m4.optim_result.stats.gevals > 0

            expected_error = ArgumentError(
                "You must provide an initial value when using generic constraints."
            )
            @test_throws expected_error maximum_likelihood(gdemo_default; cons_args...)
        end

        @testset "MAP with generic constraints" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))

            # Set two constraints: The first parameter must be non-negative, and the L2 norm
            # of the parameters must be between 0.5 and 2.
            cons(res, x, _) = (res .= [x[1], sqrt(sum(x .^ 2))])
            lcons = [0, 0.5]
            ucons = [Inf, 2.0]
            cons_args = (cons=cons, lcons=lcons, ucons=ucons)
            init_value = [0.5, -1.0]

            function check_success(result)
                optimum = result.values.array
                @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
                @test result.optim_result.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(result.lp, true_logp, atol=0.01)
            end

            m1 = estimate_mode(gdemo_default, MAP(), init_value; cons_args...)
            m2 = estimate_mode(gdemo_default, MAP(), true_value; cons_args...)
            m3 = estimate_mode(gdemo_default, MAP(), init_value, IPNewton(); cons_args...)
            m4 = estimate_mode(gdemo_default, MAP(), init_value, IPNewton(); adtype=AutoReverseDiff(), cons_args...)
            m5 = maximum_a_posteriori(gdemo_default, init_value; cons_args...)

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)

            @test m2.optim_result.stats.iterations <= 1

            @test m3.optim_result.stats.gevals > 0
            @test m4.optim_result.stats.gevals > 0

            expected_error = ArgumentError(
                "You must provide an initial value when using generic constraints."
            )
            @test_throws expected_error maximum_a_posteriori(gdemo_default; cons_args...)
        end
    end

    @testset "StatsBase integration" begin
        Random.seed!(54321)
        mle_est = maximum_likelihood(gdemo_default)
        # Calculated based on the two data points in gdemo_default, [1.5, 2.0]
        true_values = [0.0625, 1.75]

        @test coefnames(mle_est) == [:s, :m]

        diffs = coef(mle_est).array - [0.0625031; 1.75001]
        @test all(isapprox.(diffs, 0.0, atol=0.1))

        infomat = [2/(2*true_values[1]^2) 0.0; 0.0 2/true_values[1]]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        vcovmat = [2*true_values[1]^2/2 0.0; 0.0 true_values[1]/2]
        @test all(isapprox.(vcovmat - vcov(mle_est), 0.0, atol=0.01))

        ctable = coeftable(mle_est)
        @test ctable isa StatsBase.CoefTable

        s = stderror(mle_est).array
        @test all(isapprox.(s - [0.06250415643292194, 0.17677963626053916], 0.0, atol=0.01))

        @test coefnames(mle_est) == Distributions.params(mle_est)
        @test vcov(mle_est) == inv(informationmatrix(mle_est))

        @test isapprox(loglikelihood(mle_est), -0.0652883561466624, atol=0.01)
    end

    @testset "Linear regression test" begin
        @model function regtest(x, y)
            beta ~ MvNormal(Zeros(2), I)
            mu = x * beta
            y ~ MvNormal(mu, I)
        end

        Random.seed!(987)
        true_beta = [1.0, -2.2]
        x = rand(40, 2)
        y = x * true_beta

        model = regtest(x, y)
        mle = maximum_likelihood(model)

        vcmat = inv(x'x)
        vcmat_mle = vcov(mle).array

        @test isapprox(mle.values.array, true_beta)
        @test isapprox(vcmat, vcmat_mle)
    end

    @testset "Dot tilde test" begin
        @model function dot_gdemo(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))

            (.~)(x, Normal(m, sqrt(s)))
        end

        model_dot = dot_gdemo([1.5, 2.0])

        mle1 = maximum_likelihood(gdemo_default)
        mle2 = maximum_likelihood(model_dot)

        map1 = maximum_a_posteriori(gdemo_default)
        map2 = maximum_a_posteriori(model_dot)

        @test isapprox(mle1.values.array, mle2.values.array)
        @test isapprox(map1.values.array, map2.values.array)
    end

    # TODO(mhauru): The corresponding Optim.jl test had a note saying that some models
    # don't work for Tracker and ReverseDiff. Is that still the case?
    @testset "MAP for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        result_true = DynamicPPL.TestUtils.posterior_optima(model)

        optimizers = [LBFGS(), NelderMead(), LD_TNEWTON_PRECOND_RESTART()]
        @testset "$(nameof(typeof(optimizer)))" for optimizer in optimizers
            result = maximum_a_posteriori(model, optimizer)
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
                    @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol = 0.05
                end
            end
        end
    end

    # Some of the models have one variance parameter per observation, and so
    # the MLE should have the variances set to 0. Since we're working in
    # transformed space, this corresponds to `-Inf`, which is of course not achievable.
    # In particular, it can result in "early termniation" of the optimization process
    # because we hit NaNs, etc. To avoid this, we set the `g_tol` and the `f_tol` to
    # something larger than the default.
    allowed_incorrect_mle = [
        DynamicPPL.TestUtils.demo_dot_assume_dot_observe,
        DynamicPPL.TestUtils.demo_assume_index_observe,
        DynamicPPL.TestUtils.demo_assume_multivariate_observe,
        DynamicPPL.TestUtils.demo_assume_observe_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_submodel,
        DynamicPPL.TestUtils.demo_dot_assume_dot_observe_matrix,
        DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix,
        DynamicPPL.TestUtils.demo_assume_submodel_observe_index_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index_literal,
        DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix,
    ]
    @testset "MLE for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        result_true = DynamicPPL.TestUtils.likelihood_optima(model)

        optimizers = [LBFGS(), NelderMead(), LD_TNEWTON_PRECOND_RESTART()]
        @testset "$(nameof(typeof(optimizer)))" for optimizer in optimizers
            result = maximum_likelihood(model, optimizer; reltol=1e-3)
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
                    if model.f in allowed_incorrect_mle
                        @test isfinite(get(result_true, vn_leaf))
                    else
                        @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol = 0.05
                    end
                end
            end
        end
    end

    # TODO(mhauru) Make a separate test file for OptimizationCore, and move this test
    # there.
    # Issue: https://discourse.julialang.org/t/two-equivalent-conditioning-syntaxes-giving-different-likelihood-values/100320
    @testset "OptimizationContext" begin
        @model function model1(x)
            μ ~ Uniform(0, 2)
            x ~ LogNormal(μ, 1)
        end

        @model function model2()
            μ ~ Uniform(0, 2)
            x ~ LogNormal(μ, 1)
        end

        x = 1.0
        w = [1.0]

        @testset "With ConditionContext" begin
            m1 = model1(x)
            m2 = model2() | (x=x,)
            ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end

        @testset "With prefixes" begin
            function prefix_μ(model)
                return DynamicPPL.contextualize(model, DynamicPPL.PrefixContext{:inner}(model.context))
            end
            m1 = prefix_μ(model1(x))
            m2 = prefix_μ(model2() | (var"inner.x"=x,))
            ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end

        @testset "Weighted" begin
            function override(model)
                return DynamicPPL.contextualize(
                    model,
                    OverrideContext(model.context, 100, 1)
                )
            end
            m1 = override(model1(x))
            m2 = override(model2() | (x=x,))
            ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end
    end

    # Issue: https://discourse.julialang.org/t/turing-mixture-models-with-dirichlet-weightings/112910
    @testset "Optimization with different linked dimensionality" begin
        @model demo_dirichlet() = x ~ Dirichlet(2 * ones(3))
        model = demo_dirichlet()
        result = estimate_mode(model, MAP())
        @test result.values ≈ mode(Dirichlet(2 * ones(3))) atol = 0.2
    end
end
