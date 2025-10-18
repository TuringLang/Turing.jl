module OptimisationTests

using ..Models: gdemo, gdemo_default
using AbstractPPL: AbstractPPL
using Distributions
using Distributions.FillArrays: Zeros
using DynamicPPL: DynamicPPL
using LinearAlgebra: Diagonal, I
using Random: Random
using Optimization
using Optimization: Optimization
using OptimizationBBO: OptimizationBBO
using OptimizationNLopt: OptimizationNLopt
using OptimizationOptimJL: OptimizationOptimJL
using ReverseDiff: ReverseDiff
using StatsBase: StatsBase
using StatsBase: coef, coefnames, coeftable, informationmatrix, stderror, vcov
using Test: @test, @testset, @test_throws
using Turing

@testset "Optimisation" begin

    # The `stats` field is populated only in newer versions of OptimizationOptimJL and
    # similar packages. Hence we end up doing this check a lot
    hasstats(result) = result.optim_result.stats !== nothing

    # Issue: https://discourse.julialang.org/t/two-equivalent-conditioning-syntaxes-giving-different-likelihood-values/100320
    @testset "OptimLogDensity and contexts" begin
        @model function model1(x)
            μ ~ Uniform(0, 2)
            return x ~ LogNormal(μ, 1)
        end

        @model function model2()
            μ ~ Uniform(0, 2)
            return x ~ LogNormal(μ, 1)
        end

        x = 1.0
        w = [1.0]

        @testset "With ConditionContext" begin
            m1 = model1(x)
            m2 = model2() | (x=x,)
            # Doesn't matter if we use getlogjoint or getlogjoint_internal since the
            # VarInfo isn't linked.
            ld1 = Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogjoint)
            ld2 = Turing.Optimisation.OptimLogDensity(m2, DynamicPPL.getlogjoint_internal)
            @test ld1(w) == ld2(w)
        end

        @testset "With prefixes" begin
            vn = @varname(inner)
            m1 = prefix(model1(x), vn)
            m2 = prefix((model2() | (x=x,)), vn)
            ld1 = Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogjoint)
            ld2 = Turing.Optimisation.OptimLogDensity(m2, DynamicPPL.getlogjoint_internal)
            @test ld1(w) == ld2(w)
        end

        @testset "Joint, prior, and likelihood" begin
            m1 = model1(x)
            a = [0.3]
            ld_joint = Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogjoint)
            ld_prior = Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogprior)
            ld_likelihood = Turing.Optimisation.OptimLogDensity(
                m1, DynamicPPL.getloglikelihood
            )
            @test ld_joint(a) == ld_prior(a) + ld_likelihood(a)

            # test that the prior accumulator is calculating the right thing
            @test Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogprior)([0.3]) ≈
                -Distributions.logpdf(Uniform(0, 2), 0.3)
            @test Turing.Optimisation.OptimLogDensity(m1, DynamicPPL.getlogprior)([-0.3]) ≈
                -Distributions.logpdf(Uniform(0, 2), -0.3)
        end
    end

    @testset "errors on invalid model" begin
        @model function invalid_model()
            x ~ Normal()
            return x ~ Beta()
        end
        m = invalid_model()
        @test_throws ErrorException maximum_likelihood(m)
        @test_throws ErrorException maximum_a_posteriori(m)
    end

    @testset "gdemo" begin
        """
            check_success(result, true_value, true_logp, check_retcode=true)

        Check that the `result` returned by optimisation is close to the truth.
        """
        function check_optimisation_result(
            result, true_value, true_logp, check_retcode=true
        )
            optimum = result.values.array
            @test all(isapprox.(optimum - true_value, 0.0, atol=0.01))
            if check_retcode
                @test result.optim_result.retcode == Optimization.ReturnCode.Success
            end
            @test isapprox(result.lp, true_logp, atol=0.01)
        end

        @testset "MLE" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            m1 = Turing.Optimisation.estimate_mode(gdemo_default, MLE())
            m2 = maximum_likelihood(
                gdemo_default, OptimizationOptimJL.LBFGS(); initial_params=true_value
            )
            m3 = maximum_likelihood(gdemo_default, OptimizationOptimJL.Newton())
            m4 = maximum_likelihood(
                gdemo_default, OptimizationOptimJL.BFGS(); adtype=AutoReverseDiff()
            )
            m5 = maximum_likelihood(
                gdemo_default, OptimizationOptimJL.NelderMead(); initial_params=true_value
            )
            m6 = maximum_likelihood(gdemo_default, OptimizationOptimJL.NelderMead())

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            if hasstats(m6) && hasstats(m5)
                @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations
            end

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals == 0
            @test !hasstats(m6) || m6.optim_result.stats.gevals == 0
        end

        @testset "MAP" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result) = check_optimisation_result(result, true_value, true_logp)

            m1 = Turing.Optimisation.estimate_mode(gdemo_default, MAP())
            m2 = maximum_a_posteriori(
                gdemo_default, OptimizationOptimJL.LBFGS(); initial_params=true_value
            )
            m3 = maximum_a_posteriori(gdemo_default, OptimizationOptimJL.Newton())
            m4 = maximum_a_posteriori(
                gdemo_default, OptimizationOptimJL.BFGS(); adtype=AutoReverseDiff()
            )
            m5 = maximum_a_posteriori(
                gdemo_default, OptimizationOptimJL.NelderMead(); initial_params=true_value
            )
            m6 = maximum_a_posteriori(gdemo_default, OptimizationOptimJL.NelderMead())

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            if hasstats(m6) && hasstats(m5)
                @test m5.optim_result.stats.iterations < m6.optim_result.stats.iterations
            end

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals == 0
            @test !hasstats(m6) || m6.optim_result.stats.gevals == 0
        end

        @testset "MLE with box constraints" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result, check_retcode=true) =
                check_optimisation_result(result, true_value, true_logp, check_retcode)

            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            m1 = Turing.Optimisation.estimate_mode(gdemo_default, MLE(); lb=lb, ub=ub)
            m2 = maximum_likelihood(
                gdemo_default,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                initial_params=true_value,
                lb=lb,
                ub=ub,
            )
            m3 = maximum_likelihood(
                gdemo_default,
                OptimizationBBO.BBO_separable_nes();
                maxiters=100_000,
                abstol=1e-5,
                lb=lb,
                ub=ub,
            )
            m4 = maximum_likelihood(
                gdemo_default,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.BFGS());
                adtype=AutoReverseDiff(),
                lb=lb,
                ub=ub,
            )
            m5 = maximum_likelihood(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=true_value,
                lb=lb,
                ub=ub,
            )
            m6 = maximum_likelihood(gdemo_default; lb=lb, ub=ub)

            check_success(m1)
            check_success(m2)
            # BBO retcodes are misconfigured, so skip checking the retcode in this case.
            # See https://github.com/SciML/Optimization.jl/issues/745
            check_success(m3, false)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            @test !hasstats(m5) || m5.optim_result.stats.iterations <= 1

            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals == 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals > 0
        end

        @testset "MAP with box constraints" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result, check_retcode=true) =
                check_optimisation_result(result, true_value, true_logp, check_retcode)

            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            m1 = Turing.Optimisation.estimate_mode(gdemo_default, MAP(); lb=lb, ub=ub)
            m2 = maximum_a_posteriori(
                gdemo_default,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                initial_params=true_value,
                lb=lb,
                ub=ub,
            )
            m3 = maximum_a_posteriori(
                gdemo_default,
                OptimizationBBO.BBO_separable_nes();
                maxiters=100_000,
                abstol=1e-5,
                lb=lb,
                ub=ub,
            )
            m4 = maximum_a_posteriori(
                gdemo_default,
                OptimizationOptimJL.Fminbox(OptimizationOptimJL.BFGS());
                adtype=AutoReverseDiff(),
                lb=lb,
                ub=ub,
            )
            m5 = maximum_a_posteriori(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=true_value,
                lb=lb,
                ub=ub,
            )
            m6 = maximum_a_posteriori(gdemo_default; lb=lb, ub=ub)

            check_success(m1)
            check_success(m2)
            # BBO retcodes are misconfigured, so skip checking the retcode in this case.
            # See https://github.com/SciML/Optimization.jl/issues/745
            check_success(m3, false)
            check_success(m4)
            check_success(m5)
            check_success(m6)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1
            @test !hasstats(m5) || m5.optim_result.stats.iterations <= 1

            @show m2.optim_result.stats
            @test !hasstats(m2) || m2.optim_result.stats.gevals > 0
            @test !hasstats(m3) || m3.optim_result.stats.gevals == 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0
            @test !hasstats(m5) || m5.optim_result.stats.gevals > 0
        end

        @testset "MLE with generic constraints" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            true_logp = loglikelihood(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result, check_retcode=true) =
                check_optimisation_result(result, true_value, true_logp, check_retcode)

            # Set two constraints: The first parameter must be non-negative, and the L2 norm
            # of the parameters must be between 0.5 and 2.
            cons(res, x, _) = (res .= [x[1], sqrt(sum(x .^ 2))])
            lcons = [0, 0.5]
            ucons = [Inf, 2.0]
            cons_args = (cons=cons, lcons=lcons, ucons=ucons)
            initial_params = [0.5, -1.0]

            m1 = Turing.Optimisation.estimate_mode(
                gdemo_default, MLE(); initial_params=initial_params, cons_args...
            )
            m2 = maximum_likelihood(gdemo_default; initial_params=true_value, cons_args...)
            m3 = maximum_likelihood(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=initial_params,
                cons_args...,
            )
            m4 = maximum_likelihood(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=initial_params,
                adtype=AutoReverseDiff(),
                cons_args...,
            )
            m5 = maximum_likelihood(
                gdemo_default; initial_params=initial_params, cons_args...
            )

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1

            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0

            expected_error = ArgumentError(
                "You must provide an initial value when using generic constraints."
            )
            @test_throws expected_error maximum_likelihood(gdemo_default; cons_args...)
        end

        @testset "MAP with generic constraints" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            true_logp = logjoint(gdemo_default, (s=true_value[1], m=true_value[2]))
            check_success(result, check_retcode=true) =
                check_optimisation_result(result, true_value, true_logp, check_retcode)

            # Set two constraints: The first parameter must be non-negative, and the L2 norm
            # of the parameters must be between 0.5 and 2.
            cons(res, x, _) = (res .= [x[1], sqrt(sum(x .^ 2))])
            lcons = [0, 0.5]
            ucons = [Inf, 2.0]
            cons_args = (cons=cons, lcons=lcons, ucons=ucons)
            initial_params = [0.5, -1.0]

            m1 = Turing.Optimisation.estimate_mode(
                gdemo_default, MAP(); initial_params=initial_params, cons_args...
            )
            m2 = maximum_a_posteriori(
                gdemo_default; initial_params=true_value, cons_args...
            )
            m3 = maximum_a_posteriori(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=initial_params,
                cons_args...,
            )
            m4 = maximum_a_posteriori(
                gdemo_default,
                OptimizationOptimJL.IPNewton();
                initial_params=initial_params,
                adtype=AutoReverseDiff(),
                cons_args...,
            )
            m5 = maximum_a_posteriori(
                gdemo_default; initial_params=initial_params, cons_args...
            )

            check_success(m1)
            check_success(m2)
            check_success(m3)
            check_success(m4)
            check_success(m5)

            @test !hasstats(m2) || m2.optim_result.stats.iterations <= 1

            @test !hasstats(m3) || m3.optim_result.stats.gevals > 0
            @test !hasstats(m4) || m4.optim_result.stats.gevals > 0

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

        infomat = [2/(2 * true_values[1]^2) 0.0; 0.0 2/true_values[1]]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        vcovmat = [2 * true_values[1]^2/2 0.0; 0.0 true_values[1]/2]
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
            return y ~ MvNormal(mu, I)
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

            return (.~)(x, Normal(m, sqrt(s)))
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
        Random.seed!(23)
        result_true = DynamicPPL.TestUtils.posterior_optima(model)

        optimizers = [
            OptimizationOptimJL.LBFGS(),
            OptimizationOptimJL.NelderMead(),
            OptimizationNLopt.NLopt.LD_TNEWTON_PRECOND_RESTART(),
        ]
        @testset "$(nameof(typeof(optimizer)))" for optimizer in optimizers
            result = maximum_a_posteriori(model, optimizer)
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in AbstractPPL.varname_leaves(vn, get(result_true, vn))
                    @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol = 0.05
                end
            end
        end
    end

    # Some of the models have one variance parameter per observation, and so
    # the MLE should have the variances set to 0. Since we're working in
    # transformed space, this corresponds to `-Inf`, which is of course not achievable.
    # In particular, it can result in "early termination" of the optimization process
    # because we hit NaNs, etc. To avoid this, we set the `g_tol` and the `f_tol` to
    # something larger than the default.
    allowed_incorrect_mle = [
        DynamicPPL.TestUtils.demo_dot_assume_observe,
        DynamicPPL.TestUtils.demo_assume_index_observe,
        DynamicPPL.TestUtils.demo_assume_multivariate_observe,
        DynamicPPL.TestUtils.demo_assume_multivariate_observe_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_submodel,
        DynamicPPL.TestUtils.demo_dot_assume_observe_matrix_index,
        DynamicPPL.TestUtils.demo_assume_submodel_observe_index_literal,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index,
        DynamicPPL.TestUtils.demo_dot_assume_observe_index_literal,
        DynamicPPL.TestUtils.demo_assume_matrix_observe_matrix_index,
    ]
    @testset "MLE for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        Random.seed!(23)
        result_true = DynamicPPL.TestUtils.likelihood_optima(model)

        optimizers = [
            OptimizationOptimJL.LBFGS(),
            OptimizationOptimJL.NelderMead(),
            OptimizationNLopt.NLopt.LD_TNEWTON_PRECOND_RESTART(),
        ]
        @testset "$(nameof(typeof(optimizer)))" for optimizer in optimizers
            result = maximum_likelihood(model, optimizer; reltol=1e-3)
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in AbstractPPL.varname_leaves(vn, get(result_true, vn))
                    if model.f in allowed_incorrect_mle
                        @test isfinite(get(result_true, vn_leaf))
                    else
                        @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol = 0.05
                    end
                end
            end
        end
    end

    # Issue: https://discourse.julialang.org/t/turing-mixture-models-with-dirichlet-weightings/112910
    @testset "Optimization with different linked dimensionality" begin
        @model demo_dirichlet() = x ~ Dirichlet(2 * ones(3))
        model = demo_dirichlet()
        result = maximum_a_posteriori(model)
        @test result.values ≈ mode(Dirichlet(2 * ones(3))) atol = 0.2
    end

    @testset "with :=" begin
        @model function demo_track()
            x ~ Normal()
            return y := 100 + x
        end
        model = demo_track()
        result = maximum_a_posteriori(model)
        @test result.values[:x] ≈ 0 atol = 1e-1
        @test result.values[:y] ≈ 100 atol = 1e-1
    end

    @testset "get ModeResult" begin
        @model function demo_model(N)
            half_N = N ÷ 2
            a ~ arraydist(LogNormal.(fill(0, half_N), 1))
            b ~ arraydist(LogNormal.(fill(0, N - half_N), 1))
            covariance_matrix = Diagonal(vcat(a, b))
            x ~ MvNormal(covariance_matrix)
            return nothing
        end

        N = 12
        m = demo_model(N) | (x=randn(N),)
        result = maximum_a_posteriori(m)
        get_a = get(result, :a)
        get_b = get(result, :b)
        get_ab = get(result, [:a, :b])
        @assert keys(get_a) == (:a,)
        @assert keys(get_b) == (:b,)
        @assert keys(get_ab) == (:a, :b)
        @assert get_b[:b] == get_ab[:b]
        @assert vcat(get_a[:a], get_b[:b]) == result.values.array
        @assert get(result, :c) == (; :c => Array{Float64}[])
    end

    @testset "Collinear coeftable" begin
        xs = [-1.0, 0.0, 1.0]
        ys = [0.0, 0.0, 0.0]

        @model function collinear(x, y)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            return y ~ MvNormal(a .* x .+ b .* x, 1)
        end

        model = collinear(xs, ys)
        mle_estimate = Turing.Optimisation.estimate_mode(model, MLE())
        tab = coeftable(mle_estimate)
        @assert isnan(tab.cols[2][1])
        @assert tab.colnms[end] == "Error notes"
        @assert occursin("singular", tab.cols[end][1])
    end

    @testset "Negative variance" begin
        # A model for which the likelihood has a saddle point at x=0, y=0.
        # Creating an optimisation result for this model at the x=0, y=0 results in negative
        # variance for one of the variables, because the variance is calculated as the
        # diagonal of the inverse of the Hessian.
        @model function saddle_model()
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
            @addlogprob! x^2 - y^2
            return nothing
        end
        m = saddle_model()
        optim_ld = Turing.Optimisation.OptimLogDensity(m, DynamicPPL.getloglikelihood)
        vals = Turing.Optimisation.NamedArrays.NamedArray([0.0, 0.0])
        m = Turing.Optimisation.ModeResult(vals, nothing, 0.0, optim_ld)
        ct = coeftable(m)
        @assert isnan(ct.cols[2][1])
        @assert ct.colnms[end] == "Error notes"
        @assert occursin("Negative variance", ct.cols[end][1])
    end

    @testset "Same coeftable with/without numerrors_warnonly" begin
        xs = [0.0, 1.0, 2.0]

        @model function extranormal(x)
            mean ~ Normal(0, 1)
            return x ~ Normal(mean, 1)
        end

        model = extranormal(xs)
        mle_estimate = Turing.Optimisation.estimate_mode(model, MLE())
        warnonly_coeftable = coeftable(mle_estimate; numerrors_warnonly=true)
        no_warnonly_coeftable = coeftable(mle_estimate; numerrors_warnonly=false)
        @assert warnonly_coeftable.cols == no_warnonly_coeftable.cols
        @assert warnonly_coeftable.colnms == no_warnonly_coeftable.colnms
        @assert warnonly_coeftable.rownms == no_warnonly_coeftable.rownms
        @assert warnonly_coeftable.pvalcol == no_warnonly_coeftable.pvalcol
        @assert warnonly_coeftable.teststatcol == no_warnonly_coeftable.teststatcol
    end
end

end
