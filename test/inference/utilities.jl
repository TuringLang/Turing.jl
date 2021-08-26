@testset "predict" begin
    Random.seed!(100)

    @model function linear_reg(x, y, σ = 0.1)
        β ~ Normal(0, 1)

        for i ∈ eachindex(y)
            y[i] ~ Normal(β * x[i], σ)
        end
    end

    @model function linear_reg_vec(x, y, σ = 0.1)
        β ~ Normal(0, 1)
        y ~ MvNormal(β .* x, σ^2 * I)
    end

    f(x) = 2 * x + 0.1 * randn()

    Δ = 0.1
    xs_train = 0:Δ:10; ys_train = f.(xs_train);
    xs_test = [10 + Δ, 10 + 2 * Δ]; ys_test = f.(xs_test);

    # Infer
    m_lin_reg = linear_reg(xs_train, ys_train);
    chain_lin_reg = sample(m_lin_reg, NUTS(100, 0.65), 200);

    # Predict on two last indices
    m_lin_reg_test = linear_reg(xs_test, fill(missing, length(ys_test)));    
    predictions = Turing.Inference.predict(m_lin_reg_test, chain_lin_reg)

    ys_pred = vec(mean(Array(group(predictions, :y)); dims = 1))

    @test sum(abs2, ys_test - ys_pred) ≤ 0.1

    # Ensure that `rng` is respected
    predictions1 = let rng = MersenneTwister(42)
        predict(rng, m_lin_reg_test, chain_lin_reg[1:2])
    end
    predictions2 = let rng = MersenneTwister(42)
        predict(rng, m_lin_reg_test, chain_lin_reg[1:2])
    end
    @test all(Array(predictions1) .== Array(predictions2))

    # Predict on two last indices for vectorized
    m_lin_reg_test = linear_reg_vec(xs_test, missing);
    predictions_vec = Turing.Inference.predict(m_lin_reg_test, chain_lin_reg)
    ys_pred_vec = vec(mean(Array(group(predictions_vec, :y)); dims = 1))

    @test sum(abs2, ys_test - ys_pred_vec) ≤ 0.1

    # Multiple chains
    chain_lin_reg = sample(m_lin_reg, NUTS(100, 0.65), MCMCThreads(), 200, 2);
    m_lin_reg_test = linear_reg(xs_test, fill(missing, length(ys_test)));
    predictions = Turing.Inference.predict(m_lin_reg_test, chain_lin_reg)

    @test size(chain_lin_reg, 3) == size(predictions, 3)

    for chain_idx in MCMCChains.chains(chain_lin_reg)
        ys_pred = vec(mean(Array(group(predictions[:, :, chain_idx], :y)); dims = 1))
        @test sum(abs2, ys_test - ys_pred) ≤ 0.1
    end

    # Predict on two last indices for vectorized
    m_lin_reg_test = linear_reg_vec(xs_test, missing);
    predictions_vec = Turing.Inference.predict(m_lin_reg_test, chain_lin_reg)

    for chain_idx in MCMCChains.chains(chain_lin_reg)
        ys_pred_vec = vec(mean(
            Array(group(predictions_vec[:, :, chain_idx], :y));
            dims = 1
        ))
        @test sum(abs2, ys_test - ys_pred_vec) ≤ 0.1
    end

    # https://github.com/TuringLang/Turing.jl/issues/1352
    @model function simple_linear1(x, y)
        intercept ~ Normal(0,1)
        coef ~ MvNormal(zeros(2), I)
        coef = reshape(coef, 1, size(x,1))

        mu = intercept .+ coef * x |> vec
        error ~ truncated(Normal(0,1), 0, Inf)
        y ~ MvNormal(mu, error^2 * I)
    end;

    @model function simple_linear2(x, y)
        intercept ~ Normal(0,1)
        coef ~ filldist(Normal(0,1), 2)
        coef = reshape(coef, 1, size(x,1))

        mu = intercept .+ coef * x |> vec
        error ~ truncated(Normal(0,1), 0, Inf)
        y ~ MvNormal(mu, error^2 * I)
    end;

    @model function simple_linear3(x, y)
        intercept ~ Normal(0,1)
        coef = Vector(undef, 2)
        for i in axes(coef, 1)
            coef[i] ~ Normal(0,1)
        end
        coef = reshape(coef, 1, size(x,1))

        mu = intercept .+ coef * x |> vec
        error ~ truncated(Normal(0,1), 0, Inf)
        y ~ MvNormal(mu, error^2 * I)
    end;

    @model function simple_linear4(x, y)
        intercept ~ Normal(0,1)
        coef1 ~ Normal(0,1)
        coef2 ~ Normal(0,1)
        coef = [coef1, coef2]
        coef = reshape(coef, 1, size(x,1))

        mu = intercept .+ coef * x |> vec
        error ~ truncated(Normal(0,1), 0, Inf)
        y ~ MvNormal(mu, error^2 * I)
    end;

    # Some data
    x = randn(2, 100);
    y = [1 + 2 * a + 3 * b for (a,b) in eachcol(x)];

    for model in [simple_linear1, simple_linear2, simple_linear3, simple_linear4]
        m = model(x, y);
        chain = sample(m, NUTS(), 100);
        chain_predict = predict(model(x, missing), chain);
        mean_prediction = [chain_predict["y[$i]"].data |> mean for i = 1:length(y)]
        @test mean(abs2, mean_prediction - y) ≤ 1e-3
    end
end

@testset "Timer" begin
    chain = sample(gdemo_default, MH(), 1000)

    @test chain.info.start_time isa Float64
    @test chain.info.stop_time isa Float64
    @test chain.info.start_time < chain.info.stop_time
end
