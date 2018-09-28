using Turing, Test

@model perf(d) = begin
           θ ~ d
           return x
       end

# smpl = NUTS(5000,  0.65)
smpl = HMC(10000, 0.1, 5)
smpl2 = DynamicNUTS(5000)

# Standard tests for all distributions involving a single-sample.
function single_sample_tests(dist)

    # Check that invlink is inverse of link.
    x = rand(dist, 5000)
    y = sample(perf(dist), smpl)
    y2 = sample(perf(dist), smpl2)

    @test mean(y[:θ]) ≈ mean(x) atol=1e-3 # MC answer
    # @test mean(y[:θ]) ≈ mean(d) atol=1e-3 # Exact answer.
    @test mean(y2[:θ]) ≈ mean(d) atol=1e-3 # Exact answer.

end

# Scalar tests
@testset "scalar" begin
let
    # Tests with scalar-valued distributions.
    uni_dists = [
        Arcsine(2, 4),
        Beta(2,2),
        BetaPrime(),
        Biweight(),
        Cauchy(),
        Chi(3),
        Chisq(2),
        Cosine(),
        Epanechnikov(),
        Erlang(),
        Exponential(),
        FDist(1, 1),
        Frechet(),
        Gamma(),
        InverseGamma(),
        InverseGaussian(),
        # Kolmogorov(),
        Laplace(),
        Levy(),
        Logistic(),
        LogNormal(1.0, 2.5),
        Normal(0.1, 2.5),
        Pareto(),
        Rayleigh(1.0),
        TDist(2),
        TruncatedNormal(0, 1, -Inf, 2),
    ]
    for dist in uni_dists

        single_sample_tests(dist)

    end
end
end
