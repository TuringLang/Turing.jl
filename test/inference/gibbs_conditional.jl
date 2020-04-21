using Random, Turing, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")


@turing_testset "gibbs conditionals" begin
    let N = 1000,
        (α_0, θ_0) = (2.0, inv(3.0)),
        λ_true = rand(Gamma(α_0, θ_0)),
        σ_true = √(1 / λ_true),
        m_true = rand(Normal(0, σ_true)),
        x = rand(Normal(m_true, σ_true), N)
        
        @model inverse_gdemo(x) = begin
            λ ~ Gamma(α_0, θ_0)
            m ~ Normal(0, sqrt(1 / λ))
            x .~ Normal(m, sqrt(1 / λ))
        end

        function gdemo_statistics(x)
            # The conditionals and posterior can be formulated in terms of the following statistics:
            N = length(x) # number of samples
            x_bar = mean(x) # sample mean
            s2 = var(x; mean=x_bar, corrected=false) # sample variance
            return N, x_bar, s2
        end

        function cond_m(c)
            N, x_bar, s2 = gdemo_statistics(x)
            m_n = N * x_bar / (N + 1)
            λ_n = c.λ * (N + 1)
            σ_n = sqrt(1 / λ_n)
            return Normal(m_n, σ_n)
        end

        function cond_λ(c)
            N, x_bar, s2 = gdemo_statistics(x)
            α_n = α_0 + (N - 1) / 2 + 1
            β_n = s2 * N / 2 + c.m^2 / 2 + inv(θ_0)
            return Gamma(α_n, inv(β_n))
        end

        # Three tests: one for each variable fixed to the true value, and one for both
        # using the conditional
        Random.seed!(100)

        alg = Gibbs(
            GibbsConditional(:m, cond_m),
            GibbsConditional(:λ, c -> Normal(λ_true, 0)))
        chain = sample(inverse_gdemo(x), alg, 10_000)
        check_numerical(chain, [:m, :λ], [m_true, λ_true], atol=0.2)

        alg = Gibbs(
            GibbsConditional(:m, c -> Normal(m_true, 0)),
            GibbsConditional(:λ, cond_λ))
        chain = sample(inverse_gdemo(x), alg, 10_000)
        check_numerical(chain, [:m, :λ], [m_true, λ_true], atol=0.2)
        
        alg = Gibbs(
            GibbsConditional(:m, cond_m),
            GibbsConditional(:λ, cond_λ))
        chain = sample(inverse_gdemo(x), alg, 10_000)
        check_numerical(chain, [:m, :λ], [m_true, λ_true], atol=0.2)
    end

    let π = [0.5, 0.5],
        K = length(π),
        m = 0.5,
        λ = 2.0,
        σ = 0.1,
        x = [σ .* randn(10); 1 .+ σ .* randn(10)],
        N = length(x)
        
        @model mixture(x) = begin
            μ ~ arraydist(Normal.(fill(m, K), fill(λ, K)))
            z ~ arraydist(Categorical.(fill(π, N)))
            x ~ arraydist(Normal.(μ[z], σ))
            return x
        end

        # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
        function cond_z(c)
            function mixtureweight(x)
                p = π .* pdf.(Normal.(c.μ, σ), Ref(x))
                return p ./ sum(p)
            end
            return arraydist(Categorical.(mixtureweight.(x)))
        end

        function cond_μ(c)
            z = c.z
            n = [count(z .== k) for k = 1:K]

            # If there were no observations assigned to center `k`, `n[k] == 0`, and
            # we use the prior instead.
            x_bar = [(n[k] != 0) ? (sum(x[z .== k]) / n[k]) : m for k = 1:K]
            λ_hat = [(n[k] != 0) ? inv(n[k] / σ^2 + 1/λ^2) : λ for k = 1:K]
            μ_hat = [(n[k] != 0) ? x_bar[k] * (n[k] / σ^2) * λ_hat[k] : m for k = 1:K]

            return arraydist(Normal.(μ_hat, λ_hat))
        end

        μ_true = [0, 1]
        z_true = [fill(1, 10); fill(2, 10)]
        
        # Both variables sampled using the Gibbs conditional
        # We can't be sure about the order of cluster assignment, so we check both
        Random.seed!(100)
        alg = Gibbs(GibbsConditional(:z, cond_z), GibbsConditional(:μ, cond_μ))
        chain = sample(mixture(x), alg, 10000)
        μ_hat = dropdims(mean(chain[:μ].value, dims=1), dims=(1, 3))
        @test isapprox(μ_hat, μ_true, atol=0.1) || isapprox(μ_hat, reverse(μ_true), atol=0.1)
        z_hat = dropdims(mean(chain[:z].value, dims=1), dims=(1, 3))
        @test isapprox(z_hat, z_true, atol=0.2, rtol=0.0) ||
            isapprox(z_hat, z_true[[11:20; 1:10]], atol=0.2, rtol=0.0)

        # Gibbs conditional for `z`, MH for `μ`
        Random.seed!(100)
        alg = Gibbs(GibbsConditional(:z, cond_z), MH(:μ))
        chain = sample(mixture(x), alg, 10000)
        μ_hat = dropdims(mean(chain[:μ].value, dims=1), dims=(1, 3))
        @test isapprox(μ_hat, μ_true, atol=0.1) || isapprox(μ_hat, reverse(μ_true), atol=0.1)
        z_hat = dropdims(mean(chain[:z].value, dims=1), dims=(1, 3))
        @test isapprox(z_hat, z_true, atol=0.2, rtol=0.0) ||
            isapprox(z_hat, z_true[[11:20; 1:10]], atol=0.2, rtol=0.0)
        
        # Gibbs conditional for `z`, HMC for `μ`
        Random.seed!(100)
        alg = Gibbs(GibbsConditional(:z, cond_z), HMC(0.05, 4, :μ,))
        chain = sample(mixture(x), alg, 10000)
        μ_hat = dropdims(mean(chain[:μ].value, dims=1), dims=(1, 3))
        @test isapprox(μ_hat, μ_true, atol=0.1) || isapprox(μ_hat, reverse(μ_true), atol=0.1)
        z_hat = dropdims(mean(chain[:z].value, dims=1), dims=(1, 3))
        @test isapprox(z_hat, z_true, atol=0.2, rtol=0.0) ||
            isapprox(z_hat, z_true[[11:20; 1:10]], atol=0.2, rtol=0.0)
    end
end
