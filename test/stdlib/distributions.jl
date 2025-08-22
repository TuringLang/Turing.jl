module DistributionsTests

using ..NumericalTests: check_dist_numerical
using Distributions
using LinearAlgebra: I
using Random: Random
using StableRNGs: StableRNG
using StatsFuns: logistic
using Test: @testset, @test
using Turing

@testset "distributions.jl" begin
    rng = StableRNG(12345)
    @testset "distributions functions" begin
        ns = 10
        logitp = randn(rng)
        d1 = BinomialLogit(ns, logitp)
        d2 = Binomial(ns, logistic(logitp))
        k = 3
        @test logpdf(d1, k) ≈ logpdf(d2, k)
    end

    @testset "distributions functions" begin
        d = OrderedLogistic(-2, [-1, 1])

        n = 1_000_000
        y = rand(rng, d, n)
        K = length(d.cutpoints) + 1
        p = [mean(==(k), y) for k in 1:K]          # empirical probs
        pmf = [exp(logpdf(d, k)) for k in 1:K]

        @test all(((x, y),) -> abs(x - y) < 0.001, zip(p, pmf))
    end

    @testset "distribution functions" begin
        d = OrderedLogistic(0, [1, 2, 3])

        K = length(d.cutpoints) + 1
        @test support(d) == 1:K

        # Adding up probabilities sums to 1
        s = sum(pdf.(d, support(d)))
        @test s ≈ 1.0 atol = 0.0001
    end

    @testset "distributions functions" begin
        λ = 0.01:0.01:5
        LLp = @. logpdf(Poisson(λ), 1)
        LLlp = @. logpdf(LogPoisson(log(λ)), 1)
        @test LLp ≈ LLlp atol = 0.0001
    end

    @testset "single distribution correctness" begin
        rng = StableRNG(1)

        n_samples = 10_000
        mean_tol = 0.1
        var_atol = 1.0
        var_tol = 0.5
        multi_dim = 4
        # 1. UnivariateDistribution
        # NOTE: Noncentral distributions are commented out because of
        #       AD incompatibility of their logpdf functions
        dist_uni = [
            Arcsine(1, 3),
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
            # VonMises(0, 1), WARNING: this is commented are because the
            # test is broken
            Weibull(2, 1),
            # Cauchy(0, 1),  # mean and variance are undefined for Cauchy
        ]

        # 2. MultivariateDistribution
        dist_multi = [
            MvNormal(zeros(multi_dim), I),
            MvNormal(zeros(2), [2.0 1.0; 1.0 4.0]),
            Dirichlet(multi_dim, 2.0),
        ]

        # 3. MatrixDistribution
        dist_matrix = [
            Wishart(7, [1.0 0.5; 0.5 1.0]), InverseWishart(7, [1.0 0.5; 0.5 1.0])
        ]

        @testset "Correctness test for single distributions" begin
            for (dist_set, dist_list) in [
                ("UnivariateDistribution", dist_uni),
                ("MultivariateDistribution", dist_multi),
                ("MatrixDistribution", dist_matrix),
            ]
                @testset "$(string(dist_set))" begin
                    for dist in dist_list
                        @testset "$(string(typeof(dist)))" begin
                            @info "Distribution(params)" dist

                            @model m() = x ~ dist

                            chn = sample(rng, m(), HMC(0.05, 20), n_samples)

                            # Numerical tests.
                            check_dist_numerical(
                                dist,
                                chn;
                                mean_tol=mean_tol,
                                var_atol=var_atol,
                                var_tol=var_tol,
                            )
                        end
                    end
                end
            end
        end
    end
end

end
