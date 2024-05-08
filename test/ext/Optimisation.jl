@testset "ext/Optimisation.jl" begin
    @testset "gdemo" begin
        @testset "MLE" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]

            # TODO Below are the old tests, make sure we cover everything they covered.
            #f1 = optim_function(gdemo_default, MLE();constrained=false)
            #p1 = OptimizationProblem(f1.func, f1.init(true_value))

            #p2 = optim_objective(gdemo_default, MLE();constrained=false)
            #
            #p3 = optim_problem(gdemo_default, MLE();constrained=false, init_theta=true_value)

            #m1 = solve(p1, NelderMead())
            #m2 = solve(p1, LBFGS())
            #m3 = solve(p1, BFGS())
            #m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), NelderMead())
            #m5 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), LBFGS())
            #m6 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), BFGS())
            #m7 = solve(p3.prob, NelderMead())
            #m8 = solve(p3.prob, LBFGS())
            #m9 = solve(p3.prob, BFGS())

            m1 = estimate_mode(gdemo_default, MLE())
            m2 = estimate_mode(gdemo_default, MLE(), true_value, NelderMead())
            m3 = estimate_mode(gdemo_default, MLE(), BFGS())
            m4 = estimate_mode(gdemo_default, MLE(), NelderMead())

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))
        end

        @testset "MAP" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]

            # TODO Below are the old tests, make sure we cover everything they covered.
            #f1 = optim_function(gdemo_default, MAP();constrained=false)
            #p1 = OptimizationProblem(f1.func, f1.init(true_value))

            #p2 = optim_objective(gdemo_default, MAP();constrained=false)
            #
            #p3 = optim_problem(gdemo_default, MAP();constrained=false,init_theta=true_value)

            #m1 = solve(p1, NelderMead())
            #m2 = solve(p1, LBFGS())
            #m3 = solve(p1, BFGS())
            # These are commented out because p2.init is not a thing anymore. What were
            # these doing, where is optimize defined?
            # m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), NelderMead())
            # m5 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), LBFGS())
            # m6 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), p2.init(true_value), BFGS())
            #m7 = solve(p3.prob, NelderMead())
            #m8 = solve(p3.prob, LBFGS())
            #m9 = solve(p3.prob, BFGS())

            m1 = estimate_mode(gdemo_default, MAP())
            m2 = estimate_mode(gdemo_default, MAP(), true_value, NelderMead())
            m3 = estimate_mode(gdemo_default, MAP(), BFGS())
            m4 = estimate_mode(gdemo_default, MAP(), NelderMead())

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))

            #@test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(f1.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p2.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p2.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p3.transform(m7.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p3.transform(m8.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p3.transform(m9.minimizer) - true_value, 0.0, atol=0.01))
        end

        @testset "MLE constrained" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]
            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            # TODO Below are the old tests, make sure we cover everything they covered.
            #f1 = optim_function(gdemo_default, MLE();constrained=true)
            #p1 = OptimizationProblem(f1.func, f1.init(true_value); lb=lb, ub=ub)

            #p2 = optim_objective(gdemo_default, MLE();constrained=true)
            #
            #p3 = optim_problem(gdemo_default, MLE();constrained=true, init_theta=true_value, lb=lb, ub=ub)

            #m1 = solve(p1, Fminbox(LBFGS()))
            #m2 = solve(p1, Fminbox(BFGS()))
            #m3 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), lb, ub, p2.init(true_value), Fminbox(LBFGS()))
            #m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), lb, ub, p2.init(true_value), Fminbox(BFGS()))
            #m5 = solve(p3.prob, Fminbox(LBFGS()))
            #m6 = solve(p3.prob, Fminbox(BFGS()))

            m1 = estimate_mode(gdemo_default, MLE(), true_value, Fminbox(LBFGS()); lb=lb, ub=ub)
            m2 = estimate_mode(gdemo_default, MLE(), Fminbox(BFGS()); lb=lb, ub=ub)

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
        end

        @testset "MAP constrained" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]
            lb = [0.0, 0.0]
            ub = [2.0, 2.0]

            # TODO Below are the old tests, make sure we cover everything they covered.
            #f1 = optim_function(gdemo_default, MAP();constrained=true)
            #p1 = OptimizationProblem(f1.func, f1.init(true_value); lb=lb, ub=ub)

            #p2 = optim_objective(gdemo_default, MAP();constrained=true)
            #
            #p3 = optim_problem(gdemo_default, MAP();constrained=true, init_theta=true_value, lb=lb, ub=ub)

            #m1 = solve(p1, Fminbox(LBFGS()))
            #m2 = solve(p1, Fminbox(BFGS()))
            #m3 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), lb, ub, p2.init(true_value), Fminbox(LBFGS()))
            #m4 = optimize(p2.obj, (G,z) -> p2.obj(nothing,G,z), lb, ub, p2.init(true_value), Fminbox(BFGS()))
            #m5 = solve(p3.prob, Fminbox(LBFGS()))
            #m6 = solve(p3.prob, Fminbox(BFGS()))

            m1 = estimate_mode(gdemo_default, MAP(), true_value, Fminbox(LBFGS()); lb=lb, ub=ub)
            m2 = estimate_mode(gdemo_default, MAP(), Fminbox(BFGS()); lb=lb, ub=ub)

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))

            #@test all(isapprox.(f1.transform(m1.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(f1.transform(m2.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p2.transform(m3.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p2.transform(m4.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p3.transform(m5.minimizer) - true_value, 0.0, atol=0.01))
            #@test all(isapprox.(p3.transform(m6.minimizer) - true_value, 0.0, atol=0.01))
        end
    end

    @numerical_testset "Optimization.jl interface" begin
        @testset "MLE" begin
            Random.seed!(222)
            true_value = [0.0625, 1.75]

            m1 = estimate_mode(gdemo_default, MLE())
            m2 = estimate_mode(gdemo_default, MLE(), NelderMead())
            m3 = estimate_mode(gdemo_default, MLE(), true_value, LBFGS())
            m4 = estimate_mode(gdemo_default, MLE(), true_value)

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))
        end

        @testset "MAP" begin
            Random.seed!(222)
            true_value = [49 / 54, 7 / 6]

            m1 = estimate_mode(gdemo_default, MAP())
            m2 = estimate_mode(gdemo_default, MAP(), NelderMead())
            m3 = estimate_mode(gdemo_default, MAP(), true_value, LBFGS())
            m4 = estimate_mode(gdemo_default, MAP(), true_value)

            @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
            @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))
        end

        # TODO Below are tests copied over from the tests of the Optim.jl interface.
        # Make sure we implement similar tests here.

        #@testset "StatsBase integration" begin
        #    Random.seed!(54321)
        #    mle_est = estimate_mode(gdemo_default, MLE())
        #    # Calculated based on the two data points in gdemo_default, [1.5, 2.0]
        #    true_values = [0.0625, 1.75]

        #    @test coefnames(mle_est) == [:s, :m]

        #    diffs = coef(mle_est).array - [0.0625031; 1.75001]
        #    @test all(isapprox.(diffs, 0.0, atol=0.1))

        #    infomat = [2/(2 * true_values[1]^2) 0.0; 0.0 2/true_values[1]]
        #    @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        #    vcovmat = [2*true_values[1]^2 / 2 0.0; 0.0 true_values[1]/2]
        #    @test all(isapprox.(vcovmat - vcov(mle_est), 0.0, atol=0.01))

        #    ctable = coeftable(mle_est)
        #    @test ctable isa StatsBase.CoefTable

        #    s = stderror(mle_est).array
        #    @test all(isapprox.(s - [0.06250415643292194, 0.17677963626053916], 0.0, atol=0.01))

        #    @test coefnames(mle_est) == Distributions.params(mle_est)
        #    @test vcov(mle_est) == inv(informationmatrix(mle_est))

        #    @test isapprox(loglikelihood(mle_est), -0.0652883561466624, atol=0.01)
        #end

        #@testset "Linear regression test" begin
        #    @model function regtest(x, y)
        #        beta ~ MvNormal(Zeros(2), I)
        #        mu = x*beta
        #        y ~ MvNormal(mu, I)
        #    end
        #    
        #    Random.seed!(987)
        #    true_beta = [1.0, -2.2]
        #    x = rand(40, 2)
        #    y = x*true_beta
        #    
        #    model = regtest(x, y)
        #    mle = estimate_mode(model, MLE())
        #    
        #    vcmat = inv(x'x)
        #    vcmat_mle = vcov(mle).array
        #    
        #    @test isapprox(mle.values.array, true_beta)
        #    @test isapprox(vcmat, vcmat_mle)
        #end

        #@testset "Dot tilde test" begin
        #    @model function dot_gdemo(x)
        #        s ~ InverseGamma(2,3)
        #        m ~ Normal(0, sqrt(s))
        #    
        #        (.~)(x, Normal(m, sqrt(s)))
        #    end
        #    
        #    model_dot = dot_gdemo([1.5, 2.0])

        #    mle1 = estimate_mode(gdemo_default, MLE())
        #    mle2 = estimate_mode(model_dot, MLE())

        #    map1 = estimate_mode(gdemo_default, MAP())
        #    map2 = estimate_mode(model_dot, MAP())

        #    @test isapprox(mle1.values.array, mle2.values.array)
        #    @test isapprox(map1.values.array, map2.values.array)
        #end

        ## FIXME: Some models doesn't work for Tracker and ReverseDiff.
        #@testset "MAP for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        #    result_true = DynamicPPL.TestUtils.posterior_optima(model)

        #    @testset "$(nameof(typeof(optimizer)))" for optimizer in [LBFGS(), NelderMead()]
        #        result = estimate_mode(model, MAP(), optimizer)
        #        vals = result.values

        #        for vn in DynamicPPL.TestUtils.varnames(model)
        #            for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
        #                @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol=0.05
        #            end
        #        end
        #    end
        #end


        ## Some of the models have one variance parameter per observation, and so
        ## the MLE should have the variances set to 0. Since we're working in
        ## transformed space, this corresponds to `-Inf`, which is of course not achievable.
        ## In particular, it can result in "early termniation" of the optimization process
        ## because we hit NaNs, etc. To avoid this, we set the `g_tol` and the `f_tol` to
        ## something larger than the default.
        #allowed_incorrect_mle = [
        #    DynamicPPL.TestUtils.demo_dot_assume_dot_observe,
        #    DynamicPPL.TestUtils.demo_assume_index_observe,
        #    DynamicPPL.TestUtils.demo_assume_multivariate_observe,
        #    DynamicPPL.TestUtils.demo_assume_observe_literal,
        #    DynamicPPL.TestUtils.demo_dot_assume_observe_submodel,
        #    DynamicPPL.TestUtils.demo_dot_assume_dot_observe_matrix,
        #    DynamicPPL.TestUtils.demo_dot_assume_matrix_dot_observe_matrix,
        #    DynamicPPL.TestUtils.demo_assume_submodel_observe_index_literal,
        #    DynamicPPL.TestUtils.demo_dot_assume_observe_index,
        #    DynamicPPL.TestUtils.demo_dot_assume_observe_index_literal,
        #    DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix,
        #]
        #@testset "MLE for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        #    result_true = DynamicPPL.TestUtils.likelihood_optima(model)

        #    # `NelderMead` seems to struggle with convergence here, so we exclude it.
        #    @testset "$(nameof(typeof(optimizer)))" for optimizer in [LBFGS(),]
        #        result = estimate_mode(model, MLE(), optimizer, Optim.Options(g_tol=1e-3, f_tol=1e-3))
        #        vals = result.values

        #        for vn in DynamicPPL.TestUtils.varnames(model)
        #            for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
        #                if model.f in allowed_incorrect_mle
        #                    @test isfinite(get(result_true, vn_leaf))
        #                else
        #                    @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol=0.05
        #                end
        #            end
        #        end
        #    end
        #end

        ## Issue: https://discourse.julialang.org/t/two-equivalent-conditioning-syntaxes-giving-different-likelihood-values/100320
        #@testset "OptimizationContext" begin
        #    @model function model1(x)
        #        μ ~ Uniform(0, 2)
        #        x ~ LogNormal(μ, 1)
        #    end

        #    @model function model2()
        #        μ ~ Uniform(0, 2)
        #        x ~ LogNormal(μ, 1)
        #    end

        #    x = 1.0
        #    w = [1.0]

        #    @testset "With ConditionContext" begin
        #        m1 = model1(x)
        #        m2 = model2() | (x = x,)
        #        ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
        #        @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        #    end

        #    @testset "With prefixes" begin
        #        function prefix_μ(model)
        #            return DynamicPPL.contextualize(model, DynamicPPL.PrefixContext{:inner}(model.context))
        #        end
        #        m1 = prefix_μ(model1(x))
        #        m2 = prefix_μ(model2() | (var"inner.x" = x,))
        #        ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
        #        @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        #    end

        #    @testset "Weighted" begin
        #        function override(model)
        #            return DynamicPPL.contextualize(
        #                model,
        #                OverrideContext(model.context, 100, 1)
        #            )
        #        end
        #        m1 = override(model1(x))
        #        m2 = override(model2() | (x = x,))
        #        ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
        #        @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        #    end
        #end

        ## Issue: https://discourse.julialang.org/t/turing-mixture-models-with-dirichlet-weightings/112910
        #@testset "with different linked dimensionality" begin
        #    @model demo_dirichlet() = x ~ Dirichlet(2 * ones(3))
        #    model = demo_dirichlet()
        #    result = estimate_mode(model, MAP())
        #    @test result.values ≈ mode(Dirichlet(2 * ones(3))) atol=0.2
        #end
    end
end
