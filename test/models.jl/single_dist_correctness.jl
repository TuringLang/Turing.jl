using Test, Turing
turnprogress(false)

n_samples = 20_000
mean_atol = 0.25
var_atol = 1.0
multi_dim = 10

# 1. UnivariateDistribution
# NOTE: Noncentral distributions are commented out because of
#       AD imcompatibility of their logpdf functions
dist_uni = [Arcsine(1, 3),
            Beta(2, 1),
            # NoncentralBeta(2, 1, 1),
            BetaPrime(1, 1),
            Biweight(0, 1),
            Chi(7),
            Chisq(7),
            # NoncentralChisq(7, 1),
            Cosine(0, 1),
            Epanechnikov(0, 1),
            Erlang(2, 3),
            Exponential(0.1),
            FDist(7, 7),
            # NoncentralF(7, 7, 1),
            Frechet(2, 0.5),
            Normal(0, 1),
            GeneralizedExtremeValue(0, 1, 0.5),
            GeneralizedPareto(0, 1, 0.5),
            Gumbel(0, 0.5),
            InverseGaussian(1, 1),
            Kolmogorov(),
            # KSDist(2),  # no pdf function defined
            # KSOneSided(2),  # no pdf function defined
            Laplace(0, 0.5),
            Levy(0, 1),
            Logistic(0, 1),
            LogNormal(0, 1),
            Gamma(2, 3),
            InverseGamma(3, 1),
            NormalCanon(0, 1),
            NormalInverseGaussian(0, 2, 1, 1),
            Pareto(1, 1),
            Rayleigh(1),
            SymTriangularDist(0, 1),
            TDist(2.5),
            # NoncentralT(2.5, 1),
            TriangularDist(1, 3, 2),
            Triweight(0, 1),
            Uniform(0, 1),
            VonMises(0, 1),
            Weibull(2, 1),
            # Cauchy(0, 1),  # mean and variance are undefined for Cauchy
           ]

# 2. MultivariateDistribution
dist_multi = [MvNormal(zeros(multi_dim), ones(multi_dim)),
              MvNormal(zeros(2), [2 1; 1 4]),
              Dirichlet(multi_dim, 2.0),
             ]

# 3. MatrixDistribution
dist_matrix = [Wishart(7, [1 0.5; 0.5 1]),
               InverseWishart(7, [1 0.5; 0.5 1]),
              ]

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
