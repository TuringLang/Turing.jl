using Test, Turing
turnprogress(false)

n_samples = 20_000
mean_atol = 0.25
var_atol = 1.0
multi_dim = 10

# 1. UnivariateDistribution
dist_uni = [Normal(0, 1),
            TDist(2.5),
            Gamma(2, 3),
            InverseGamma(3, 1),
            Beta(2, 1),
            #Cauchy(0, 1),       # mean and variance are undefined for Cauchy
            ]

# 2. MultivariateDistribution
dist_multi = [MvNormal(zeros(multi_dim), ones(multi_dim)),
              MvNormal(zeros(2), [2 1; 1 4]),
              Dirichlet(multi_dim, 2.0),
              ]

# 3. MatrixDistribution
dist_matrix = [Wishart(7, [1 0.5; 0.5 1])]

@testset "Correctness test for single distributions" begin
    for (dist_set, dist_list) ∈ [("UnivariateDistribution",   dist_uni),
                                 ("MultivariateDistribution", dist_multi),
                                 ("MatrixDistribution",       dist_matrix)
                                 ]
        @testset "$(string(dist_set))" begin
            for dist in dist_list
                @testset "$(string(typeof(dist)))" begin
                    @info "Distribution(params)" dist

                    @model m() = begin
                        x ~ dist
                    end
                    chn = sample(m(), NUTS(n_samples, 0.8))

                    chn_xs = chn[:x][1:2:end] # thining by halving

                    # Mean
                    dist_mean = mean(dist)
                    if !all(isnan.(dist_mean)) && !all(isinf.(dist_mean))
                        chn_mean = mean(chn_xs)
                        @test chn_mean ≈ dist_mean atol=(mean_atol * length(chn_mean))
                    end

                    # var() for Distributions.MatrixDistribution is not defined
                    if !(dist isa Distributions.MatrixDistribution)
                        # Variance
                        dist_var = var(dist)
                        if !all(isnan.(dist_var)) && !all(isinf.(dist_var))
                            chn_var = var(chn_xs)
                            @test chn_var ≈ dist_var atol=(var_atol * length(chn_var))
                        end
                    end
                end
            end
        end
    end
end
# Wishart(7, [1 0.5; 0.5 1])
