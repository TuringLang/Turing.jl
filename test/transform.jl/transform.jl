using Turing, Test
using Turing: link, invlink, logpdf_with_trans
using ForwardDiff: derivative, jacobian
using LinearAlgebra: logabsdet

# logabsdet doesn't handle scalars.
_logabsdet(x::AbstractArray) = logabsdet(x)[1]
_logabsdet(x::Real) = log(abs(x))

# Generate a (vector / matrix of) random number(s).
_rand_real(::Real) = randn()
_rand_real(x) = randn(size(x))

# Standard tests for all distributions involving a single-sample.
function single_sample_tests(dist, jacobian)

    # Do the regular single-sample tests.
    single_sample_tests(dist)

    # Check that the implementation of the logpdf agrees with the AD version.
    x = rand(dist)
    logpdf_ad = logpdf(dist, x) - _logabsdet(jacobian(x->link(dist, x), x))
    @test logpdf_ad ≈ logpdf_with_trans(dist, x, true)
end

# Standard tests for all distributions involving a single-sample. Doesn't check that the
# logpdf implementation is consistent with the link function for technical reasons.
function single_sample_tests(dist)

    # Check that invlink is inverse of link.
    x = rand(dist)
    @test invlink(dist, link(dist, copy(x))) ≈ x atol=1e-9

    # Check that link is inverse of invlink. Hopefully this just holds given the above...
    y = link(dist, x)
    @test link(dist, invlink(dist, copy(y))) ≈ y atol=1e-9

    # This should probably be exact.
    @test logpdf(dist, x) == logpdf_with_trans(dist, x, false)

    # Check that invlink maps back to the apppropriate constrained domain.
    @test all(isfinite, logpdf.(Ref(dist), [invlink(dist, _rand_real(x)) for _ in 1:100]))

    # This is a quirk of the current implementation, of which it would be nice to be rid.
    @test typeof(x) == typeof(y)
end

# Standard tests for all distributions involving multiple samples. xs should be whatever
# the appropriate repeated version of x is for the distribution in question. ie. for
# univariate distributions, just a vector of identical values. For vector-valued
# distributions, a matrix whose columns are identical.
function multi_sample_tests(dist, x, xs, N)
    ys = link(dist, copy(xs))
    @test invlink(dist, link(dist, copy(xs))) ≈ xs atol=1e-9
    @test link(dist, invlink(dist, copy(ys))) ≈ ys atol=1e-9
    @test logpdf_with_trans(dist, xs, true) == fill(logpdf_with_trans(dist, x, true), N)
    @test logpdf_with_trans(dist, xs, false) == fill(logpdf_with_trans(dist, x, false), N)

    # This is a quirk of the current implementation, of which it would be nice to be rid.
    @test typeof(xs) == typeof(ys)
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

        single_sample_tests(dist, derivative)

        # specialised multi-sample tests.
        N = 10
        x = rand(dist)
        xs = fill(x, N)
        multi_sample_tests(dist, x, xs, N)
    end
end
end

# Tests with vector-valued distributions.
@testset "vector" begin
let
    vector_dists = [
        Dirichlet(2, 3),
        MvNormal(randn(10), exp.(randn(10))),
        MvLogNormal(MvNormal(randn(10), exp.(randn(10)))),
    ]
    for dist in vector_dists

        if dist isa Dirichlet
            single_sample_tests(dist)

            # This should fail at the minute. Not sure what the correct way to test this is.
            x = rand(dist)
            logpdf_turing = logpdf_with_trans(dist, x, true)
            J = jacobian(x->link(dist, x), x)
            @test_broken logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing
        else
            single_sample_tests(dist, jacobian)
        end

        # Multi-sample tests. Columns are observations due to Distributions.jl conventions.
        N = 10
        x = rand(dist)
        xs = repeat(x, 1, N)
        multi_sample_tests(dist, x, xs, N)
    end
end
end

# Tests with matrix-valued distributions.
@testset "matrix" begin
let
    matrix_dists = [
        Wishart(7, [1 0.5; 0.5 1]),
        InverseWishart(2, [1 0.5; 0.5 1]),
    ]
    for dist in matrix_dists

        single_sample_tests(dist)

        # This should fail at the minute. Not sure what the correct way to test this is.
        x = rand(dist)
        logpdf_turing = logpdf_with_trans(dist, x, true)
        J = jacobian(x->link(dist, x), x)
        @test_broken logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing

        # Multi-sample tests comprising vectors of matrices.
        N = 10
        x = rand(dist)
        xs = [x for _ in 1:N]
        multi_sample_tests(dist, x, xs, N)
    end
end
end

################################## Miscelaneous old tests ##################################

# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), exp.([-1000., -1000., -1000.]), true)
# NaN
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), [-1000., -1000., -1000.], true, true)
# -1999.30685281944
#
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), exp.([-1., -2., -3.]), true)
# -3.006450206744678
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), [-1., -2., -3.], true, true)
# -3.006450206744678
d  = Dirichlet([1., 1., 1.])
r  = [-1000., -1000., -1000.]
r2 = [-1., -2., -3.]

# test link
#link(d, r)

# test invlink
@test invlink(d, r) ≈ [0., 0., 1.] atol=1e-9

# test logpdf_with_trans
#@test logpdf_with_trans(d, invlink(d, r), true) -1999.30685281944 1e-9 ≈ # atol=NaN
@test logpdf_with_trans(d, invlink(d, r2), true) ≈ -3.760398892580863 atol=1e-9
