@numerical_testset "ext/OptimInterface.jl" begin
    @testset "MLE" begin
        Random.seed!(222)
        true_value = [0.0625, 1.75]

        m1 = optimize(gdemo_default, MLE())
        m2 = optimize(gdemo_default, MLE(), NelderMead())
        m3 = optimize(gdemo_default, MLE(), true_value, LBFGS())
        m4 = optimize(gdemo_default, MLE(), true_value)

        @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))
    end

    @testset "MAP" begin
        Random.seed!(222)
        true_value = [49 / 54, 7 / 6]

        m1 = optimize(gdemo_default, MAP())
        m2 = optimize(gdemo_default, MAP(), NelderMead())
        m3 = optimize(gdemo_default, MAP(), true_value, LBFGS())
        m4 = optimize(gdemo_default, MAP(), true_value)

        @test all(isapprox.(m1.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m2.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m3.values.array - true_value, 0.0, atol=0.01))
        @test all(isapprox.(m4.values.array - true_value, 0.0, atol=0.01))
    end

    @testset "StatsBase integration" begin
        Random.seed!(54321)
        mle_est = optimize(gdemo_default, MLE())
        # Calculated based on the two data points in gdemo_default, [1.5, 2.0]
        true_values = [0.0625, 1.75]

        @test coefnames(mle_est) == [:s, :m]

        diffs = coef(mle_est).array - [0.0625031; 1.75001]
        @test all(isapprox.(diffs, 0.0, atol=0.1))

        infomat = [2/(2 * true_values[1]^2) 0.0; 0.0 2/true_values[1]]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        vcovmat = [2*true_values[1]^2 / 2 0.0; 0.0 true_values[1]/2]
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
            mu = x*beta
            y ~ MvNormal(mu, I)
        end
        
        Random.seed!(987)
        true_beta = [1.0, -2.2]
        x = rand(40, 2)
        y = x*true_beta
        
        model = regtest(x, y)
        mle = optimize(model, MLE())
        
        vcmat = inv(x'x)
        vcmat_mle = vcov(mle).array
        
        @test isapprox(mle.values.array, true_beta)
        @test isapprox(vcmat, vcmat_mle)
    end

    @testset "Dot tilde test" begin
        @model function dot_gdemo(x)
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
        
            (.~)(x, Normal(m, sqrt(s)))
        end
        
        model_dot = dot_gdemo([1.5, 2.0])

        mle1 = optimize(gdemo_default, MLE())
        mle2 = optimize(model_dot, MLE())

        map1 = optimize(gdemo_default, MAP())
        map2 = optimize(model_dot, MAP())

        @test isapprox(mle1.values.array, mle2.values.array)
        @test isapprox(map1.values.array, map2.values.array)
    end

    # FIXME: Some models doesn't work for Tracker and ReverseDiff.
    @testset "MAP for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        result_true = DynamicPPL.TestUtils.posterior_optima(model)

        @testset "$(nameof(typeof(optimizer)))" for optimizer in [LBFGS(), NelderMead()]
            result = optimize(model, MAP(), optimizer)
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
                    @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol=0.05
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

        # `NelderMead` seems to struggle with convergence here, so we exclude it.
        @testset "$(nameof(typeof(optimizer)))" for optimizer in [LBFGS(),]
            result = optimize(model, MLE(), optimizer, Optim.Options(g_tol=1e-3, f_tol=1e-3))
            vals = result.values

            for vn in DynamicPPL.TestUtils.varnames(model)
                for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(result_true, vn))
                    if model.f in allowed_incorrect_mle
                        @test isfinite(get(result_true, vn_leaf))
                    else
                        @test get(result_true, vn_leaf) ≈ vals[Symbol(vn_leaf)] atol=0.05
                    end
                end
            end
        end
    end

    # Issue: https://discourse.julialang.org/t/turing-mixture-models-with-dirichlet-weightings/112910
    @testset "with different linked dimensionality" begin
        @model demo_dirichlet() = x ~ Dirichlet(2 * ones(3))
        model = demo_dirichlet()
        result = optimize(model, MAP())
        @test result.values ≈ mode(Dirichlet(2 * ones(3))) atol=0.2
    end
end
