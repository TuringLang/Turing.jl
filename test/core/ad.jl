using ForwardDiff, Distributions, FDM, Tracker, Random, LinearAlgebra, PDMats
using Turing: gradient_logp_reverse, invlink, link, SampleFromPrior
using Turing.Core.RandomVariables: getval
using Turing.Core: TuringMvNormal, TuringDiagNormal
using ForwardDiff: Dual
using StatsFuns: binomlogpdf, logsumexp
using Test

include("../test_utils/AllUtils.jl")

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
        rng, N = MersenneTwister(123456), 13
        B = randn(rng, N, N)

        logdet_func(B) = logdet(cholesky(B' * B + Matrix(I, N, N)))

        f_tracker, back = Tracker.forward(logdet_func, B)
        tracker_grad = back(1.0)[1]
        fdm_grad = FDM.j′vp(central_fdm(5, 1), logdet_func, 1.0, B)

        @test logdet_func(B) == f_tracker
        @test fdm_grad ≈ tracker_grad
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

        logpdf_func(m, B, x) = logpdf(MvNormal(m, B' * B + Matrix(I, N, N)), x)

        # Ensure that forward-pass is correct.
        @test Tracker.forward(logpdf_func, m, B, x)[1] ≈ logpdf_func(m, B, x)

        # Check reverse-pass with finite differencing.
        fdm_grads = FDM.j′vp(central_fdm(5, 1), logpdf_func, 1.0, m, B, x)
        out, back = Tracker.forward(logpdf_func, m, B, x)
        tracker_grads = back(1)

        @test fdm_grads[1] ≈ tracker_grads[1]
        @test fdm_grads[2] ≈ tracker_grads[2]
        @test fdm_grads[3] ≈ tracker_grads[3]
    end
    @numerical_testset "Tracker + Diagonal Normal" begin
        rng, N = MersenneTwister(123456), 11
        m, σ = randn(rng, N), exp.(0.1 .* randn(rng, N)) .+ 1

        d = TuringDiagNormal(m, σ)
        x = rand(d)

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, σ)
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        out, _ = Tracker.forward((m, σ, x)->logpdf(MvNormal(m, σ), x), m, σ, x)
        @test out ≈ logpdf(MvNormal(m, σ), x)

        diag_logpdf_func(m, σ, x) = logpdf(MvNormal(m, σ), x)

        fdm_grads = FDM.j′vp(forward_fdm(5, 1), diag_logpdf_func, 1.0, m, σ, x)
        out, back = Tracker.forward(diag_logpdf_func, m, σ, x)
        tracker_grads = back(1)

        @test fdm_grads[1] ≈ tracker_grads[1]
        @test fdm_grads[2] ≈ tracker_grads[2]
        @test fdm_grads[3] ≈ tracker_grads[3]
    end
end
