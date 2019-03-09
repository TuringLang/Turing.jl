using Test, ForwardDiff, Distributions, FDM, Flux.Tracker, Turing
using StatsFuns: binomlogpdf

# Real

x_real = randn(5)

dists = [Normal(0, 1)]

for dist in dists

    f(x::Vector) = sum(logpdf.(Ref(dist), x))

    ForwardDiff.gradient(f, x_real)

end

# Postive

x_positive = randn(5).^2

dists = [Gamma(2, 3)]

for dist in dists

    f(x::Vector) = sum(logpdf.(Ref(dist), x))

    g = x -> ForwardDiff.gradient(f, x)

end

@testset "Binomial" begin
    @testset "Binomial(10, p)(3)" begin
        foo = p->binomlogpdf(10, p, 3)
        @test isapprox(
            Tracker.gradient(foo, 0.5)[1],
            central_fdm(5, 1)(foo, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(foo, 0.5)[1],
            ForwardDiff.derivative(foo, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )

        bar = p->logpdf(Binomial(10, p), 3)
        @test isapprox(
            Tracker.gradient(bar, 0.5)[1],
            central_fdm(5, 1)(bar, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(bar, 0.5)[1],
            ForwardDiff.derivative(bar, 0.5),
            rtol=1e-8,
            atol=1e-8,
        )
    end
end

@testset "Poisson" begin
    foo = p->Turing.poislogpdf(p, 1)
    @test isapprox(
        Tracker.gradient(foo, 0.5)[1],
        central_fdm(5, 1)(foo, 0.5);
        rtol=1e-8,
        atol=1e-8,
    )
    @test isapprox(
        Tracker.gradient(foo, 0.5)[1],
        ForwardDiff.derivative(foo, 0.5);
        rtol=1e-8,
        atol=1e-8,
    )

    bar = p->logpdf(Poisson(p), 3)
    @test isapprox(
        Tracker.gradient(bar, 0.5)[1],
        central_fdm(5, 1)(bar, 0.5);
        rtol=1e-8,
        atol=1e-8,
    )
    @test isapprox(
        Tracker.gradient(bar, 0.5)[1],
        ForwardDiff.derivative(bar, 0.5);
        rtol=1e-8,
        atol=1e-8,
    )
end

@testset "NegativeBinomial" begin
    @testset "NegativeBinomial(5, p)(1)" begin
        foo = p->Turing.nbinomlogpdf(5, p, 1)
        @test isapprox(
            Tracker.gradient(foo, 0.5)[1],
            central_fdm(5, 1)(foo, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(foo, 0.5)[1],
            ForwardDiff.derivative(foo, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )

        bar = p->logpdf(NegativeBinomial(5, p), 3)
        @test isapprox(
            Tracker.gradient(bar, 0.5)[1],
            central_fdm(5, 1)(bar, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(bar, 0.5)[1],
            ForwardDiff.derivative(bar, 0.5);
            rtol=1e-8,
            atol=1e-8,
        )
    end
    @testset "NegativeBinomial(r, 0.5)(1)" begin
        foo = r->Turing.nbinomlogpdf(r, 0.5, 1)
        @test isapprox(
            Tracker.gradient(foo, 3.5)[1],
            central_fdm(5, 1)(foo, 3.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(foo, 3.5)[1],
            ForwardDiff.derivative(foo, 3.5);
            rtol=1e-8,
            atol=1e-8,
        )

        bar = r->logpdf(NegativeBinomial(r, 0.5), 3)
        @test isapprox(
            Tracker.gradient(bar, 3.5)[1],
            central_fdm(5, 1)(bar, 3.5);
            rtol=1e-8,
            atol=1e-8,
        )
        @test isapprox(
            Tracker.gradient(bar, 3.5)[1],
            ForwardDiff.derivative(bar, 3.5);
            rtol=1e-8,
            atol=1e-8,
        )
    end
    @testset "NegativeBinomial(r, p)(1)" begin
        foo = x -> Turing.nbinomlogpdf(x[1], x[2], 1)
        @test isapprox(
            Tracker.gradient(foo, [3.5, 0.5])[1],
            ForwardDiff.gradient(foo, [3.5, 0.5]);
            rtol=1e-8,
            atol=1e-8,
        )
    end
end


"""
    DistSpec{Td, Tθ, Tx}

Distribution `d` with parameters `θ` and valid member of domain of RV `x`.
"""
struct DistSpec{Td, Tθ<:Tuple, Tx}
    name::String
    d::Td
    θ::Tθ
    x::Tx
end

const GEV = GeneralizedExtremeValue

# Construct a vector of `DistSpec`s that has broad coverage of the
# continuous univariate distributions in Distributions.jl.
function continuous_univariate_dist_specs()
    return [
        DistSpec("Arcsine(1, 3)(1.2)", Arcsine, (1.0, 3.0), 1.2),
        DistSpec("Beta(2, 1)(0.3)", Beta, (2.0, 1.0), 0.3),
        DistSpec("BetaPrime(1, 1)(0.33)", BetaPrime, (1.0, 1.0), 0.33),
        DistSpec("BiWeight(0, 1)(0.4)", Biweight, (0.0, 1.0), 0.4),
        DistSpec("Cauchy(0.11, 0.93)(0.2)", Cauchy, (0.11, 0.93), 0.2),
        DistSpec("Chi(7)(0.2)", Chi, (7.0,), 0.2),
        DistSpec("Chisq(7)(0.23)", Chisq, (7.0,), 0.23),
        DistSpec("Cosine(0, 1)(0.54)", Cosine, (0.0, 1.0), 0.54),
        DistSpec("Epanechnikov(0, 1)(-0.43)", Epanechnikov, (0.0, 1.0), -0.43),
        DistSpec("Erlang(2, 3)(0.43)", θ->Erlang(2, θ), (3.0,), 0.43),
        DistSpec("Exponential(1.2)(0.12)", Exponential, (1.2,), 0.12),
        DistSpec("FDist(3, 1)(0.35)", FDist, (3.0, 1.0), 0.35),
        DistSpec("Frechet(2, 0.5)(1.1)", Frechet, (2.0, 0.5), 1.1),
        DistSpec("Gamma(2, 3)(0.45)", Gamma, (2.0, 3.0), 0.45),
        DistSpec("GEV(0.13, 1.43, 0.5)(2.3)", GEV, (0.13, 1.43, 0.5), 2.3),
        DistSpec("Normal(0.5, 1.01)(-0.25)", Normal, (0.5, 1.01), -0.25),
    ]
end

#             GeneralizedPareto(0, 1, 0.5),
#             Gumbel(0, 0.5),
#             InverseGaussian(1, 1),
#             Kolmogorov(),
#             # KSDist(2),  # no pdf function defined
#             # KSOneSided(2),  # no pdf function defined
#             Laplace(0, 0.5),
#             Levy(0, 1),
#             Logistic(0, 1),
#             LogNormal(0, 1),
#             Gamma(2, 3),
#             InverseGamma(3, 1),
#             NormalCanon(0, 1),
#             NormalInverseGaussian(0, 2, 1, 1),
#             Pareto(1, 1),
#             Rayleigh(1),
#             SymTriangularDist(0, 1),
#             TDist(2.5),
#             # NoncentralT(2.5, 1),
#             TriangularDist(1, 3, 2),
#             Triweight(0, 1),
#             Uniform(0, 1),
#             # VonMises(0, 1), WARNING: this is commented are because the test is broken
#             Weibull(2, 1),

@testset "Univariate Continuous" begin
    fdm = central_fdm(5, 1)
    dist_specs = continuous_univariate_dist_specs()
    for dist_spec in dist_specs
        @testset "$(dist_spec.name)" begin
            d, θ, x = dist_spec.d, dist_spec.θ, dist_spec.x

            f_θ = θ->logpdf(dist_spec.d(θ...), x)
            @test isapprox(
                Tracker.gradient(f_θ, [θ...])[1],
                FDM.grad(fdm, f_θ, [θ...]);
                rtol=1e-8,
                atol=1e-8,
            )

            f_x = x->logpdf(dist_spec.d(θ...), x)
            @test isapprox(
                Tracker.gradient(f_x, x)[1],
                fdm(f_x, x),
                rtol=1e-8,
                atol=1e-8,
            )
        end
    end
end



function discrete_univariate_dist_specs()
    return [
        DistSpec("Bernoulli(0.45)(1)", Bernoulli, (0.45,), 1),
        DistSpec("Bernoulli(0.45)(0)", Bernoulli, (0.45,), 0),
        DistSpec("BetaBinomial(10, 2, 1)(5)", (α, β)->BetaBinomial(10, α, β), (2.0, 1.0), 5),
        DistSpec("Binomial(10, 0.5)(5)", p->Binomial(10, p), (0.5,), 5),
        DistSpec("Categorical([0.45, 0.55])(1)", Categorical, (0.45, 0.55), 1),
        DistSpec("Geometric(0.45)(3)", Geometric, (0.45,), 3),
        DistSpec("NegativeBinomial(3.5, 0.5)(1)", NegativeBinomial, (3.5, 0.5), 1),
        DistSpec("Poisson(0.5)(1)", Poisson, (0.5,), 1),
        DistSpec("PoissonBinomial(0.5)(3)", PoissonBinomial, (0.5,), 3),
        DistSpec("Skellam(1.1, 1,2)(-2)", Skellam, (1.1, 1.2), -2),
    ]
end

@testset "Univariate Discrete" begin
    fdm = central_fdm(5, 1)
    dist_specs = discrete_univariate_dist_specs()
    for dist_spec in dist_specs
        @testset "$(dist_spec.name)" begin
            d, θ, x = dist_spec.d, dist_spec.θ, dist_spec.x

            f_θ = θ->logpdf(dist_spec.d(θ...), x)
            @test isapprox(
                Tracker.gradient(f_θ, [θ...])[1],
                FDM.grad(fdm, f_θ, [θ...]);
                rtol=1e-8,
                atol=1e-8,
            )
        end
    end
end
