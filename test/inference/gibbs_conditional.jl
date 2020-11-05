using Random, Turing, Test
using Clustering

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")


@turing_testset "gibbs conditionals" begin
    let N = 1000,
        (α_0, θ_0) = (2.0, inv(3.0)),
        λ_true = rand(Gamma(α_0, θ_0)),
        σ_true = sqrt(1 / λ_true),
        m_true = rand(Normal(0, σ_true)),
        x = rand(Normal(m_true, σ_true), N)
        
        @model function inverse_gdemo(x)
            λ ~ Gamma(α_0, θ_0)
            m ~ Normal(0, sqrt(1 / λ))
            @. x ~ $(Normal(m, sqrt(1 / λ)))
        end

        function gdemo_statistics(x)
            # The conditionals and posterior can be formulated in terms of the following statistics:
            n = length(x) # number of samples
            x_bar = mean(x) # sample mean
            s2 = var(x; mean=x_bar, corrected=false) # sample variance
            return n, x_bar, s2
        end

        function cond_m(c)
            n, x_bar, s2 = gdemo_statistics(x)
            m_n = N * x_bar / (n + 1)
            λ_n = c.λ * (n + 1)
            σ_n = sqrt(1 / λ_n)
            return Normal(m_n, σ_n)
        end

        function cond_λ(c)
            n, x_bar, s2 = gdemo_statistics(x)
            α_n = α_0 + (n - 1) / 2 + 1
            β_n = s2 * n / 2 + c.m^2 / 2 + inv(θ_0)
            return Gamma(α_n, inv(β_n))
        end

        # Three tests: one for each variable fixed to the true value, and one for both
        # using the conditional
        Random.seed!(100)
        for alg in (Gibbs(GibbsConditional(:m, cond_m),
                          GibbsConditional(:λ, c -> Normal(λ_true, 0))),
                    Gibbs(GibbsConditional(:m, c -> Normal(m_true, 0)),
                          GibbsConditional(:λ, cond_λ)),
                    Gibbs(GibbsConditional(:m, cond_m),
                          GibbsConditional(:λ, cond_λ)))
            chain = sample(inverse_gdemo(x), alg, 10_000)
            check_numerical(chain, [:m, :λ], [m_true, λ_true], atol=0.2)
        end
    end

    Random.seed!(100)
    let π = [0.5, 0.5],
        K = length(π),
        m = 0.5,
        λ = 2.0,
        σ = 0.1,
        N = 20,
        μ_true = [rand(Normal(m, λ)) for _ in π],
        z_true = [rand(Categorical(π)) for _ in 1:N],
        x = [rand(Normal(μ_true[z_n], σ)) for z_n in z_true]
        
        @model function mixture(x)
            μ ~ arraydist(Normal.(fill(m, K), fill(λ, K)))
            z ~ arraydist(Categorical.(fill(π, N)))
            x ~ arraydist(Normal.(μ[z], σ))
            return x
        end

        # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
        function cond_z(c)
            function mixtureweight(x)
                p = π .* pdf.(Normal.(c.μ, σ), x)
                return p ./ sum(p)
            end
            return arraydist(Categorical.(mixtureweight.(x)))
        end

        function cond_μ(c)
            z = c.z
            n = [count(z .== k) for k = 1:K]

            # If there were no observations assigned to center `k`, `n[k] == 0`, and
            # we use the prior instead.
            λ_hat = [(n[k] != 0) ? inv(n[k] / σ^2 + 1/λ^2) : λ for k = 1:K]
            μ_hat = [(n[k] != 0) ? (sum(x[z .== k]) / σ^2) * λ_hat[k] : m for k = 1:K]

            return arraydist(Normal.(μ_hat, λ_hat))
        end

        estimate(chain, var) = dropdims(mean(Array(group(chain, var)), dims=1), dims=1)
        function estimatez(chain, var, range)
            z = Int.(Array(group(chain, var)))
            return map(i -> findmax(counts(z[:,i], range))[2], 1:size(z,2))
        end        
        
        # Both variables sampled using the Gibbs conditional
        Random.seed!(100)
        for alg in (Gibbs(GibbsConditional(:z, cond_z), GibbsConditional(:μ, cond_μ)),
                    Gibbs(GibbsConditional(:z, cond_z), MH(:μ)),
                    Gibbs(GibbsConditional(:z, cond_z), HMC(0.01, 7, :μ)), )

            chain = sample(mixture(x), alg, 10000)
            
            μ_hat = estimate(chain, :μ)
            lμ_hat, uμ_hat = minimum(μ_hat), maximum(μ_hat)
            lμ_true, uμ_true = minimum(μ_true), maximum(μ_true)
            @test isapprox([lμ_true, uμ_true], [lμ_hat, uμ_hat], atol=0.1)
            
            z_hat = estimatez(chain, :z, 1:2)
            ari, _, _, _ = randindex(z_true, Int.(z_hat))
            @test isapprox(ari, 1, atol=0.1)
        end
    end
end
