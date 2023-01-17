function find_map(model::DynamicPPL.TestUtils.DemoModels)
    # Set up.
    true_values = rand(NamedTuple, model)
    d = length(true_values.s)
    s_size, m_size = size(true_values.s), size(true_values.m)
    s_isunivariate = true_values.s isa Real
    m_isunivariate = true_values.m isa Real

    # Cosntruct callable.
    function f_wrapped(x)
        s = s_isunivariate ? x[1] : reshape(x[1:d], s_size)
        m = m_isunivariate ? x[2] : reshape(x[d + 1:end], m_size)
        return -DynamicPPL.TestUtils.logjoint_true(model, s, m)
    end

    # Optimize.
    lbs = vcat(fill(0, d), fill(-Inf, d))
    ubs = fill(Inf, 2d)
    result = optimize(f_wrapped, lbs, ubs, rand(2d), Fminbox(NelderMead()))
    @assert Optim.converged(result) "optimization didn't converge"

    # Extract the result.
    x = Optim.minimizer(result)
    s = s_isunivariate ? x[1] : reshape(x[1:d], s_size)
    m = m_isunivariate ? x[2] : reshape(x[d + 1:end], m_size)
    return -Optim.minimum(result), (s = s, m = m)
end

@testset "OptimInterface.jl" begin
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

        @test coefnames(mle_est) == [:s, :m]

        diffs = coef(mle_est).array - [0.0625031; 1.75001]
        @test all(isapprox.(diffs, 0.0, atol=0.1))

        infomat = [0.003907027690416608 4.157954948417027e-7; 4.157954948417027e-7 0.03125155528962335]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        ctable = coeftable(mle_est)
        @test ctable isa StatsBase.CoefTable

        s = stderror(mle_est).array
        @test all(isapprox.(s - [0.06250415643292194, 0.17677963626053916], 0.0, atol=0.01))

        @test coefnames(mle_est) == Distributions.params(mle_est)
        @test vcov(mle_est) == informationmatrix(mle_est)

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
        vcmat_mle = informationmatrix(mle).array
        
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
    if Turing.Essential.ADBACKEND[] === :forwarddiff
        @testset "MAP for $(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            maximum_true, maximizer_true = find_map(model)

            @testset "$(optimizer)" for optimizer in [LBFGS(), NelderMead()]
                result = optimize(model, MAP(), optimizer)
                vals = result.values

                for vn in DynamicPPL.TestUtils.varnames(model)
                    for vn_leaf in DynamicPPL.TestUtils.varname_leaves(vn, get(maximizer_true, vn))
                        sym = DynamicPPL.AbstractPPL.getsym(vn_leaf)
                        true_value_vn = get(maximizer_true, vn_leaf)
                        @test vals[Symbol(vn_leaf)] ≈ true_value_vn rtol = 0.05
                    end
                end
            end
        end
    end
end
