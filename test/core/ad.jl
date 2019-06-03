using ForwardDiff, Distributions, FDM, Tracker, Random, LinearAlgebra, PDMats
using Turing: Turing, gradient_logp_reverse, invlink, link, SampleFromPrior
using Turing.Core.RandomVariables: getval
using Turing.Core: TuringMvNormal, TuringDiagNormal
using ForwardDiff: Dual
using StatsFuns: binomlogpdf, logsumexp
using Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

_to_cov(B) = B * B' + Matrix(I, size(B)...)

@testset "ad.jl" begin
    @turing_testset "AD compatibility" begin

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

        # Test AD.
        test_ad(p->binomlogpdf(10, p, 3))
        test_ad(p->logpdf(Binomial(10, p), 3))
        test_ad(p->Turing.poislogpdf(p, 1))
        test_ad(p->logpdf(Poisson(p), 3))
        test_ad(p->Turing.nbinomlogpdf(5, p, 1))
        test_ad(p->logpdf(NegativeBinomial(5, p), 3))
        test_ad(p->Turing.nbinomlogpdf(p, 0.5, 1), 3.5)
        test_ad(r->logpdf(NegativeBinomial(r, 0.5), 3), 3.5)
        test_ad(x->Turing.nbinomlogpdf(x[1], x[2], 1), [3.5, 0.5])
        test_ad(m->logpdf(MvNormal(m, 1.0), [1.0, 1.0]), [1.0, 1.0])
        test_ad(ms->logpdf(MvNormal(ms[1:2], ms[3]), [1.0, 1.0]), [1.0, 1.0, 1.0])
        test_ad(s->logpdf(MvNormal(zeros(2), s), [1.0, 1.0]), [1.0, 1.0])
        test_ad(ms->logpdf(MvNormal(ms[1:2], ms[3:4]), [1.0, 1.0]), [1.0, 1.0, 1.0, 1.0])
        s = rand(2,2); s = s' * s
        test_ad(m->logpdf(MvNormal(m, s), [1.0, 1.0]), [1.0, 1.0])
        test_ad(s->logpdf(MvNormal(zeros(2), s), [1.0, 1.0]), s)
        ms = [[0.0, 0.0]; s[:]]
        test_ad(ms->logpdf(MvNormal(ms[1:2], reshape(ms[3:end], 2, 2)), [1.0, 1.0]), ms)
        test_ad(logsumexp, [1.0, 1.0])
    end
    @turing_testset "adr" begin
        ad_test_f = gdemo_default
        vi = Turing.VarInfo()
        ad_test_f(vi, SampleFromPrior())
        svn = collect(Iterators.filter(vn -> vn.sym == :s, keys(vi)))[1]
        mvn = collect(Iterators.filter(vn -> vn.sym == :m, keys(vi)))[1]
        _s = getval(vi, svn)[1]
        _m = getval(vi, mvn)[1]

        x = map(x->Float64(x), vi[SampleFromPrior()])
        ∇E = gradient_logp_reverse(x, vi, ad_test_f)[2]
        grad_Turing = sort(∇E)

        dist_s = InverseGamma(2,3)

        # Hand-written logp
        function logp(x::Vector)
          s = x[2]
          # s = invlink(dist_s, s)
          m = x[1]
          lik_dist = Normal(m, sqrt(s))
          lp = logpdf(dist_s, s) + logpdf(Normal(0,sqrt(s)), m)
          lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
          lp
        end

        # Call ForwardDiff's AD
        g = x -> ForwardDiff.gradient(logp, x);
        # _s = link(dist_s, _s)
        _x = [_m, _s]
        grad_FWAD = sort(g(_x))

        # Compare result
        @test grad_Turing ≈ grad_FWAD atol=1e-9
    end
    @turing_testset "passing duals to distributions" begin
        float1 = 1.1
        float2 = 2.3
        d1 = Dual(1.1)
        d2 = Dual(2.3)

        @test logpdf(Normal(0, 1), d1).value ≈ logpdf(Normal(0, 1), float1) atol=0.001
        @test logpdf(Gamma(2, 3), d2).value ≈ logpdf(Gamma(2, 3), float2) atol=0.001
        @test logpdf(Beta(2, 3), (d2 - d1) / 2).value ≈ logpdf(Beta(2, 3), (float2 - float1) / 2) atol=0.001

        @test pdf(Normal(0, 1), d1).value ≈ pdf(Normal(0, 1), float1) atol=0.001
        @test pdf(Gamma(2, 3), d2).value ≈ pdf(Gamma(2, 3), float2) atol=0.001
        @test pdf(Beta(2, 3), (d2 - d1) / 2).value ≈ pdf(Beta(2, 3), (float2 - float1) / 2) atol=0.001
    end
    @numerical_testset "general AD tests" begin
        # Tests gdemo gradient.
        function logp1(x::Vector)
            dist_s = InverseGamma(2, 3)
            s = x[2]
            m = x[1]
            lik_dist = Normal(m, sqrt(s))
            lp = Turing.logpdf_with_trans(dist_s, s, false) + Turing.logpdf_with_trans(Normal(0,sqrt(s)), m, false)
            lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
            return lp
        end

        test_model_ad(gdemo_default, logp1, [:m, :s])

        # Test Wishart AD.
        @model wishart_ad() = begin
            v ~ Wishart(7, [1 0.5; 0.5 1])
            v
        end

        # Hand-written logp
        function logp3(x)
            dist_v = Wishart(7, [1 0.5; 0.5 1])
            v = [x[1] x[3]; x[2] x[4]]
            lp = Turing.logpdf_with_trans(dist_v, v, false)
            return lp
        end

        test_model_ad(wishart_ad(), logp3, [:v])
    end
    @numerical_testset "Tracker + logdet" begin
        rng, N = MersenneTwister(123456), 7
        ȳ, B = randn(rng), randn(rng, N, N)
        test_tracker_ad(B->logdet(cholesky(_to_cov(B))), ȳ, B; rtol=1e-8, atol=1e-8)
    end
    @numerical_testset "Tracker + fill" begin
        rng = MersenneTwister(123456)
        test_tracker_ad(x->fill(x, 7), randn(rng, 7), randn(rng))
        test_tracker_ad(x->fill(x, 7, 11), randn(rng, 7, 11), randn(rng))
        test_tracker_ad(x->fill(x, 7, 11, 13), rand(rng, 7, 11, 13), randn(rng))
    end
    @numerical_testset "Tracker + MvNormal" begin
        rng, N = MersenneTwister(123456), 11
        B = randn(rng, N, N)
        m, A = randn(rng, N), B' * B + I

        # Generate from the TuringMvNormal
        d, back = Tracker.forward(TuringMvNormal, m, A)
        x = Tracker.data(rand(d))

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, PDMat(A))
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_tracker_ad((m, B, x)->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), m, B, x)
    end
    @numerical_testset "Tracker + Diagonal Normal" begin
        rng, N = MersenneTwister(123456), 11
        m, σ = randn(rng, N), exp.(0.1 .* randn(rng, N)) .+ 1

        d = TuringDiagNormal(m, σ)
        x = rand(d)

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, σ)
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_tracker_ad((m, σ, x)->logpdf(MvNormal(m, σ), x), randn(rng), m, σ, x)
    end
    @numerical_testset "Tracker + MvNormal Interface" begin
        # Note that we only test methods where the `MvNormal` ctor actually constructs
        # a TuringMvNormal.

        rng, N = MersenneTwister(123456), 7
        m, b, B, x = randn(rng, N), randn(rng, N), randn(rng, N, N), randn(rng, N)
        ȳ = randn(rng)

        # zero mean, dense covariance
        test_tracker_ad((B, x)->logpdf(MvNormal(_to_cov(B)), x), randn(rng), B, x)
        test_tracker_ad(B->logpdf(MvNormal(_to_cov(B)), x), randn(rng), B)

        # zero mean, diagonal covariance
        test_tracker_ad((b, x)->logpdf(MvNormal(exp.(b)), x), randn(rng), b, x)
        test_tracker_ad(b->logpdf(MvNormal(exp.(b)), x), randn(rng), b)

        # dense mean, dense covariance
        test_tracker_ad((m, B, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N, N), randn(rng, N),
        )
        test_tracker_ad((m, B)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N, N),
        )
        test_tracker_ad((m, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((B, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N, N), randn(rng, N),
        )
        test_tracker_ad(m->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), randn(rng, N))
        test_tracker_ad(B->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), randn(rng, N, N))

        # dense mean, diagonal covariance
        test_tracker_ad((m, b, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((m, b)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((m, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((b, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad(m->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N),
        )
        test_tracker_ad(b->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N),
        )

        # dense mean, diagonal variance
        test_tracker_ad((m, b, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((m, b)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((m, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((b, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad(m->logpdf(MvNormal(m, exp.(b)), x), randn(rng), randn(rng, N))
        test_tracker_ad(b->logpdf(MvNormal(m, exp.(b)), x), randn(rng), randn(rng, N))

        # dense mean, constant covariance
        b_s = randn(rng)
        test_tracker_ad((m, b, x)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng, N), randn(rng), randn(rng, N),
        )
        test_tracker_ad((m, b)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng, N), randn(rng),
        )
        test_tracker_ad((m, x)->logpdf(MvNormal(m, exp(b_s)), x),
            randn(rng),
            randn(rng, N), randn(rng, N)
        )
        test_tracker_ad((b, x)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_tracker_ad(m->logpdf(MvNormal(m, exp(b_s)), x), randn(rng), randn(rng, N))
        test_tracker_ad(b->logpdf(MvNormal(m, exp(b)), x), randn(rng), randn(rng))

        # dense mean, constant variance
        b_s = randn(rng)
        test_tracker_ad((m, b, x)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng, N), randn(rng), randn(rng, N),
        )
        test_tracker_ad((m, b)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng, N), randn(rng),
        )
        test_tracker_ad((m, x)->logpdf(MvNormal(m, exp(b_s) * I), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_tracker_ad((b, x)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_tracker_ad(m->logpdf(MvNormal(m, exp(b_s) * I), x), randn(rng), randn(rng, N))
        test_tracker_ad(b->logpdf(MvNormal(m, exp(b) * I), x), randn(rng), randn(rng))

        # zero mean, constant variance
        test_tracker_ad((b, x)->logpdf(MvNormal(N, exp(b)), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_tracker_ad(b->logpdf(MvNormal(N, exp(b)), x), randn(rng), randn(rng))
    end
end
