using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@testset "hmc.jl" begin
    @numerical_testset "constrained bounded" begin
        obs = [0,1,0,1,1,1,1,1,1,1]

        @model constrained_test(obs) = begin
            p ~ Beta(2,2)
            for i = 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            p
        end

        chain = sample(
            constrained_test(obs),
            HMC(1000, 1.5, 3)) # using a large step size (1.5)

        check_numerical(chain, [:p], [10/14], eps=0.1)
    end
    @numerical_testset "contrained simplex" begin
        obs12 = [1,2,1,2,2,2,2,2,2,2]

        @model constrained_simplex_test(obs12) = begin
            ps ~ Dirichlet(2, 3)
            for i = 1:length(obs12)
                obs12[i] ~ Categorical(ps)
            end
            return ps
        end

        chain = sample(
            constrained_simplex_test(obs12),
            HMC(1000, 0.75, 2))

        check_numerical(chain, ["ps[1]", "ps[2]"], [5/16, 11/16], eps=0.015)
    end
    @numerical_testset "hmc reverse diff" begin
        alg = HMC(3000, 0.15, 10)
        res = sample(gdemo_default, alg)
        check_gdemo(res, eps=0.1)
    end
    @turing_testset "matrix support" begin
        @model hmcmatrixsup() = begin
            v ~ Wishart(7, [1 0.5; 0.5 1])
            v
        end

        model_f = hmcmatrixsup()
        vs = []
        chain = nothing
        τ = 3000
        for _ in 1:5
            chain = sample(model_f, HMC(τ, 0.1, 3))
            r = reshape(chain[:v].value, τ, 2, 2)
            push!(vs, reshape(mean(r, dims = [1]), 2, 2))
        end

        @test maximum(abs, mean(vs) - (7 * [1 0.5; 0.5 1])) <= 0.5
    end
    @turing_testset "multivariate support" begin
        function sigmoid(t)
            return 1 / (1 + exp.(-t))
        end

        # Define NN flow
        function nn(x, b1, w11, w12, w13, bo, wo)
            h = tanh.([w11' * x + b1[1]; w12' * x + b1[2]; w13' * x + b1[3]])
            return sigmoid((wo' * h)[1] + bo)
        end

        # Generating training data
        N = 20
        M = round(Int64, N / 4)
        x1s = rand(M) * 5
        x2s = rand(M) * 5
        xt1s = Array([[x1s[i]; x2s[i]] for i = 1:M])
        append!(xt1s, Array([[x1s[i] - 6; x2s[i] - 6] for i = 1:M]))
        xt0s = Array([[x1s[i]; x2s[i] - 6] for i = 1:M])
        append!(xt0s, Array([[x1s[i] - 6; x2s[i]] for i = 1:M]))

        xs = [xt1s; xt0s]
        ts = [ones(M); ones(M); zeros(M); zeros(M)]

        # Define model

        alpha = 0.16            # regularizatin term
        var = sqrt(1.0 / alpha) # variance of the Gaussian prior

        @model bnn(ts) = begin
            b1 ~ MvNormal([0 ;0; 0],
                [var 0 0; 0 var 0; 0 0 var])
            w11 ~ MvNormal([0; 0], [var 0; 0 var])
            w12 ~ MvNormal([0; 0], [var 0; 0 var])
            w13 ~ MvNormal([0; 0], [var 0; 0 var])
            bo ~ Normal(0, var)

            wo ~ MvNormal([0; 0; 0],
                [var 0 0; 0 var 0; 0 0 var])
            for i = rand(1:N, 10)
                y = nn(xs[i], b1, w11, w12, w13, bo, wo)
                ts[i] ~ Bernoulli(y)
            end
            b1, w11, w12, w13, bo, wo
        end

        # Sampling
        chain = sample(bnn(ts), HMC(10, 0.1, 5))
    end
end
