using Test, Turing
turnprogress(false)

n_samples = 20_000
mean_atol = 0.25
var_atol = 1.0
multi_dim = 10

@testset "Correctness test for single distributions" begin

for dist ∈ [Normal(0, 1),
            Gamma(2, 3),
            InverseGamma(3, 4),
            Beta(2, 1),
            #Cauchy(0, 1),       # mean and variance are undefined for Cauchy
            MvNormal(zeros(multi_dim), ones(multi_dim)),
            MvNormal(zeros(2), [2 1; 1 4]),
            # Dirichlet(multi_dim, 2.0),
            Wishart(7, [1 0.5; 0.5 1])
            ]

    @testset "$(string(dist))" begin

        @model m() = begin
            x ~ dist
        end
        mf = m()

        chn = sample(mf, NUTS(n_samples, 0.8))
        chn_xs = chn[:x][1:2:end] # thining by halving

        # Mean
        dist_mean = mean(dist)
        if !all(isnan.(collect(dist_mean))) && !all(isinf.(collect(dist_mean)))
            chn_mean = mean(chn_xs)
            @test chn_mean ≈ dist_mean atol=mean_atol*length(chn_mean)
        end

        # Variance
        dist_var = var(dist)
        if !all(isnan.(collect(dist_var))) && !all(isinf.(collect(dist_var)))
            chn_var = var(chn_xs)
            @test chn_var ≈ dist_var atol=var_atol*length(chn_var)
        end

    end

end

end

# Wishart(7, [1 0.5; 0.5 1])
