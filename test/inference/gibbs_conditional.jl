using Random, Turing, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")


@turing_testset "gibbs conditionals" begin
    let α_0 = 2.0,
        θ_0 = inv(3.0),
        x = [1.5, 2.0]
        
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
            α_n = α_0 + (N - 1) / 2
            β_n = (s2 * N / 2 + c.m^2 / 2 + inv(θ_0))
            return Gamma(α_n, inv(β_n))
        end

        Random.seed!(100)
        alg = Gibbs(
            GibbsConditional(:m, cond_m),
            GibbsConditional(:λ, cond_λ))
        chain = sample(inverse_gdemo(x), alg, 3000)
        check_numerical(chain, [:m, :λ], [7/6, 24/49], atol=0.1)
    end

    let π = [0.5, 0.5],
        K = length(π),
        m = 0.0,
        λ = 5.0,
        σ = 1.0,
        
        x = [rand(10); 2 .+ rand(10)],
        N = length(x)
        
        @model mixture(x) = begin
            μ ~ arraydist(Normal.(fill(m, K), fill(λ, K)))
            z ~ arraydist(Categorical.(fill(π, N)))
            x ~ arraydist(Normal.(μ[z], σ))
            return x
        end

        # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
        function cond_z(c)
            μ = c.μ
            logπ = log.(π)
            
            function mixtureweight(x)
                p = exp.(logπ .+ logpdf.(Normal.(μ, σ), Ref(x)))
                return p / sum(p)
            end
            return arraydist([Categorical(mixtureweight(x[n])) for n = 1:N])
        end

        function cond_μ(c)
            z = c.z
            n = [count(z .== k) for k = 1:K]

            # If there were no observations assigned to center `k`, `n[k] == 0`, and
            # we use the prior instead.
            x_bar = [(n[k] != 0) ? (sum(x[z .== k]) / n[k]) : m for k = 1:K]
            λ_hat = [(n[k] != 0) ? inv(n[k] / σ^2 + 1/λ^2) : λ for k = 1:K]
            μ_hat = [x_bar[k] * (n[k]/σ^2) * λ_hat[k] for k = 1:K]
            
            return MvNormal(μ_hat, λ_hat)
        end

        Random.seed!(100)
        alg = Gibbs(
            GibbsConditional(:z, cond_z),
            GibbsConditional(:μ, cond_μ))
        chain = sample(mixture(x), alg, 10000)
        check_numerical(chain, [:z, :μ], [[fill(1, 10); fill(2, 10)], [0.0, 2.5]], atol=0.1)
    end
end
