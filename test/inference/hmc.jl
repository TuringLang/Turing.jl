@testset "hmc.jl" begin
    @numerical_testset "constrained bounded" begin
        # Set a seed
        Random.seed!(5)
        
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
            HMC(1.5, 3),# using a large step size (1.5)
            1000)

        check_numerical(chain, [:p], [10/14], atol=0.1)
    end
    @numerical_testset "contrained simplex" begin
        obs12 = [1,2,1,2,2,2,2,2,2,2]

        @model constrained_simplex_test(obs12) = begin
            ps ~ Dirichlet(2, 3)
            pd ~ Dirichlet(4, 1)
            for i = 1:length(obs12)
                obs12[i] ~ Categorical(ps)
            end
            return ps
        end

        chain = sample(
            constrained_simplex_test(obs12),
            HMC(0.75, 2),
            1000)

        check_numerical(chain, ["ps[1]", "ps[2]"], [5/16, 11/16], atol=0.015)
    end
    @numerical_testset "hmc reverse diff" begin
        Random.seed!(1)
        alg = HMC(0.1, 10)
        res = sample(gdemo_default, alg, 4000)
        check_gdemo(res, rtol=0.1)
    end
    @turing_testset "matrix support" begin
        @model hmcmatrixsup() = begin
            v ~ Wishart(7, [1 0.5; 0.5 1])
        end

        model_f = hmcmatrixsup()
        n_samples = 1_000
        vs = map(1:3) do _
            chain = sample(model_f, HMC(0.15, 7), n_samples)
            r = reshape(Array(group(chain, :v)), n_samples, 2, 2)
            reshape(mean(r; dims = 1), 2, 2)
        end

        @test maximum(abs, mean(vs) - (7 * [1 0.5; 0.5 1])) <= 0.5
    end
    @turing_testset "multivariate support" begin
        # Define NN flow
        function nn(x, b1, w11, w12, w13, bo, wo)
            h = tanh.([w11 w12 w13]' * x .+ b1)
            return logistic(dot(wo, h) + bo)
        end

        # Generating training data
        N = 20
        M = N ÷ 4
        x1s = rand(M) * 5
        x2s = rand(M) * 5
        xt1s = Array([[x1s[i]; x2s[i]] for i = 1:M])
        append!(xt1s, Array([[x1s[i] - 6; x2s[i] - 6] for i = 1:M]))
        xt0s = Array([[x1s[i]; x2s[i] - 6] for i = 1:M])
        append!(xt0s, Array([[x1s[i] - 6; x2s[i]] for i = 1:M]))

        xs = [xt1s; xt0s]
        ts = [ones(M); ones(M); zeros(M); zeros(M)]

        # Define model

        alpha = 0.16                  # regularizatin term
        var_prior = sqrt(1.0 / alpha) # variance of the Gaussian prior

        @model bnn(ts) = begin
            b1 ~ MvNormal([0. ;0.; 0.],
                [var_prior 0. 0.; 0. var_prior 0.; 0. 0. var_prior])
            w11 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            w12 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            w13 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            bo ~ Normal(0, var_prior)

            wo ~ MvNormal([0.; 0; 0],
                [var_prior 0. 0.; 0. var_prior 0.; 0. 0. var_prior])
            for i = rand(1:N, 10)
                y = nn(xs[i], b1, w11, w12, w13, bo, wo)
                ts[i] ~ Bernoulli(y)
            end
            b1, w11, w12, w13, bo, wo
        end

        # Sampling
        chain = sample(bnn(ts), HMC(0.1, 5), 10)
    end
    Random.seed!(123)
    @numerical_testset "hmcda inference" begin
        alg1 = HMCDA(1000, 0.8, 0.015)
        # alg2 = Gibbs(HMCDA(200, 0.8, 0.35, :m), HMC(0.25, 3, :s))
        alg3 = Gibbs(PG(10, :s), HMCDA(200, 0.8, 0.005, :m))
        # alg3 = Gibbs(HMC(0.25, 3, :m), PG(30, 3, :s))
        # alg3 = PG(50, 2000)

        res1 = sample(gdemo_default, alg1, 3000)
        check_gdemo(res1)

        # res2 = sample(gdemo([1.5, 2.0]), alg2)
        #
        # @test mean(res2[:s]) ≈ 49/24 atol=0.2
        # @test mean(res2[:m]) ≈ 7/6 atol=0.2

        res3 = sample(gdemo_default, alg3, 1000)
        check_gdemo(res3)
    end
    @turing_testset "hmcda constructor" begin
        alg = HMCDA(0.8, 0.75)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75, :s)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        @test isa(alg, HMCDA)
        @test isa(sampler, Sampler{<:Turing.Hamiltonian})
    end
    @numerical_testset "nuts inference" begin
        alg = NUTS(1000, 0.8)
        res = sample(gdemo_default, alg, 6000)
        check_gdemo(res)
    end
    @turing_testset "nuts constructor" begin
        alg = NUTS(200, 0.65)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(0.65)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(200, 0.65, :m)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"
    end
    @turing_testset "check discard" begin
        alg = NUTS(100, 0.8)

        c1 = sample(gdemo_default, alg, 500, discard_adapt = true)
        c2 = sample(gdemo_default, alg, 500, discard_adapt = false)

        @test size(c1, 1) == 500
        @test size(c2, 1) == 500
    end
    @turing_testset "AHMC resize" begin
        alg1 = Gibbs(PG(10, :m), NUTS(100, 0.65, :s))
        alg2 = Gibbs(PG(10, :m), HMC(0.1, 3, :s))
        alg3 = Gibbs(PG(10, :m), HMCDA(100, 0.65, 0.3, :s))
        @test sample(gdemo_default, alg1, 300) isa Chains
        @test sample(gdemo_default, alg2, 300) isa Chains
        @test sample(gdemo_default, alg3, 300) isa Chains
    end

    @turing_testset "Regression tests" begin
        # https://github.com/TuringLang/DynamicPPL.jl/issues/27
        @model function mwe(::Type{T}=Float64) where {T<:Real}
            m = Matrix{T}(undef, 2, 3)
            @. m ~ MvNormal(zeros(2), 1)
        end
        @test sample(mwe(), HMC(0.2, 4), 1_000) isa Chains

        @model function mwe(::Type{T} = Matrix{Float64}) where T
            m = T(undef, 2, 3)
            @. m ~ MvNormal(zeros(2), 1)
        end
        @test sample(mwe(), HMC(0.2, 4), 1_000) isa Chains

        # https://github.com/TuringLang/Turing.jl/issues/1308
        @model function mwe(::Type{T} = Array{Float64}) where T
            m = T(undef, 2, 3)
            @. m ~ MvNormal(zeros(2), 1)
        end
        @test sample(mwe(), HMC(0.2, 4), 1_000) isa Chains
    end
end
