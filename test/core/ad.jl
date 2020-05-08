using ForwardDiff, Distributions, FiniteDifferences, Tracker, Random, LinearAlgebra
using PDMats, Zygote
using Turing: Turing, invlink, link, SampleFromPrior, 
    TrackerAD, ZygoteAD
using DynamicPPL: getval
using Turing.Core: TuringDenseMvNormal, TuringDiagMvNormal
using ForwardDiff: Dual
using StatsFuns: binomlogpdf, logsumexp
using Test, LinearAlgebra
const FDM = FiniteDifferences

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

_to_cov(B) = B * B' + Matrix(I, size(B)...)
@testset "ad.jl" begin
    @turing_testset "adr" begin
        ad_test_f = gdemo_default
        vi = Turing.VarInfo(ad_test_f)
        ad_test_f(vi, SampleFromPrior())
        svn = vi.metadata.s.vns[1]
        mvn = vi.metadata.m.vns[1]
        _s = getval(vi, svn)[1]
        _m = getval(vi, mvn)[1]

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

        x = map(x->Float64(x), vi[SampleFromPrior()])
        ∇E1 = gradient_logp(TrackerAD(), x, vi, ad_test_f)[2]
        @test sort(∇E1) ≈ grad_FWAD atol=1e-9

        ∇E2 = gradient_logp(ZygoteAD(), x, vi, ad_test_f)[2]
        @test sort(∇E2) ≈ grad_FWAD atol=1e-9
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
    @turing_testset "general AD tests" begin
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
    @turing_testset "Tracker, Zygote and ReverseDiff + logdet" begin
        rng, N = MersenneTwister(123456), 7
        ȳ, B = randn(rng), randn(rng, N, N)
        test_reverse_mode_ad(B->logdet(cholesky(_to_cov(B))), ȳ, B; rtol=1e-8, atol=1e-6)
    end
    @turing_testset "Tracker & Zygote + fill" begin
        rng = MersenneTwister(123456)
        test_reverse_mode_ad(x->fill(x, 7), randn(rng, 7), randn(rng))
        test_reverse_mode_ad(x->fill(x, 7, 11), randn(rng, 7, 11), randn(rng))
        test_reverse_mode_ad(x->fill(x, 7, 11, 13), rand(rng, 7, 11, 13), randn(rng))
    end
    @turing_testset "Tracker, Zygote and ReverseDiff + MvNormal" begin
        rng, N = MersenneTwister(123456), 11
        B = randn(rng, N, N)
        m, A = randn(rng, N), B' * B + I

        # Generate from the TuringDenseMvNormal
        d, back = Tracker.forward(TuringDenseMvNormal, m, A)
        x = Tracker.data(rand(d))

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, PDMat(A))
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_reverse_mode_ad((m, B, x)->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), m, B, x)
    end
    @turing_testset "Tracker, Zygote and ReverseDiff + Diagonal Normal" begin
        rng, N = MersenneTwister(123456), 11
        m, σ = randn(rng, N), exp.(0.1 .* randn(rng, N)) .+ 1

        d = TuringDiagMvNormal(m, σ)
        x = rand(d)

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, σ)
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_reverse_mode_ad((m, σ, x)->logpdf(MvNormal(m, σ), x), randn(rng), m, σ, x)
    end
    @turing_testset "Tracker, Zygote and ReverseDiff + MvNormal Interface" begin
        # Note that we only test methods where the `MvNormal` ctor actually constructs
        # a TuringDenseMvNormal.

        rng, N = MersenneTwister(123456), 7
        m, b, B, x = randn(rng, N), randn(rng, N), randn(rng, N, N), randn(rng, N)
        ȳ = randn(rng)

        # zero mean, dense covariance
        test_reverse_mode_ad((B, x)->logpdf(MvNormal(_to_cov(B)), x), randn(rng), B, x)
        test_reverse_mode_ad(B->logpdf(MvNormal(_to_cov(B)), x), randn(rng), B)

        # zero mean, diagonal covariance
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(exp.(b)), x), randn(rng), b, x)
        test_reverse_mode_ad(b->logpdf(MvNormal(exp.(b)), x), randn(rng), b)

        # dense mean, dense covariance
        test_reverse_mode_ad((m, B, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N, N), randn(rng, N),
        )
        test_reverse_mode_ad((m, B)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N, N),
        )
        test_reverse_mode_ad((m, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((B, x)->logpdf(MvNormal(m, _to_cov(B)), x),
            randn(rng),
            randn(rng, N, N), randn(rng, N),
        )
        test_reverse_mode_ad(m->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), randn(rng, N))
        test_reverse_mode_ad(B->logpdf(MvNormal(m, _to_cov(B)), x), randn(rng), randn(rng, N, N))

        # dense mean, diagonal covariance
        test_reverse_mode_ad((m, b, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((m, b)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((m, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad(m->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N),
        )
        test_reverse_mode_ad(b->logpdf(MvNormal(m, Diagonal(exp.(b))), x),
            randn(rng),
            randn(rng, N),
        )

        # dense mean, diagonal variance
        test_reverse_mode_ad((m, b, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((m, b)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((m, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(m, exp.(b)), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad(m->logpdf(MvNormal(m, exp.(b)), x), randn(rng), randn(rng, N))
        test_reverse_mode_ad(b->logpdf(MvNormal(m, exp.(b)), x), randn(rng), randn(rng, N))

        # dense mean, constant covariance
        b_s = randn(rng)
        test_reverse_mode_ad((m, b, x)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng, N), randn(rng), randn(rng, N),
        )
        test_reverse_mode_ad((m, b)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng, N), randn(rng),
        )
        test_reverse_mode_ad((m, x)->logpdf(MvNormal(m, exp(b_s)), x),
            randn(rng),
            randn(rng, N), randn(rng, N)
        )
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(m, exp(b)), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_reverse_mode_ad(m->logpdf(MvNormal(m, exp(b_s)), x), randn(rng), randn(rng, N))
        test_reverse_mode_ad(b->logpdf(MvNormal(m, exp(b)), x), randn(rng), randn(rng))

        # dense mean, constant variance
        b_s = randn(rng)
        test_reverse_mode_ad((m, b, x)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng, N), randn(rng), randn(rng, N),
        )
        test_reverse_mode_ad((m, b)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng, N), randn(rng),
        )
        test_reverse_mode_ad((m, x)->logpdf(MvNormal(m, exp(b_s) * I), x),
            randn(rng),
            randn(rng, N), randn(rng, N),
        )
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(m, exp(b) * I), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_reverse_mode_ad(m->logpdf(MvNormal(m, exp(b_s) * I), x), randn(rng), randn(rng, N))
        test_reverse_mode_ad(b->logpdf(MvNormal(m, exp(b) * I), x), randn(rng), randn(rng))

        # zero mean, constant variance
        test_reverse_mode_ad((b, x)->logpdf(MvNormal(N, exp(b)), x),
            randn(rng),
            randn(rng), randn(rng, N),
        )
        test_reverse_mode_ad(b->logpdf(MvNormal(N, exp(b)), x), randn(rng), randn(rng))
    end
    @testset "Simplex Tracker, Zygote and ReverseDiff (with and without caching) AD" begin
        @model dir() = begin
            theta ~ Dirichlet(1 ./ fill(4, 4))
        end
        Turing.setadbackend(:tracker)
        sample(dir(), HMC(0.01, 1), 1000);
        Turing.setadbackend(:zygote)
        sample(dir(), HMC(0.01, 1), 1000);
        Turing.setadbackend(:reversediff)
        Turing.setrdcache(false)
        sample(dir(), HMC(0.01, 1), 1000);
        Turing.setrdcache(true)
        sample(dir(), HMC(0.01, 1), 1000);
        @test length(Memoization.caches) == 1
        Turing.emptyrdcache()
        @test length(Memoization.caches) == 0
    end
    # FIXME: For some reasons PDMatDistribution AD tests fail with ReverseDiff
    @testset "PDMatDistribution AD" begin
        @model wishart() = begin
            theta ~ Wishart(4, Matrix{Float64}(I, 4, 4))
        end
        Turing.setadbackend(:tracker)
        sample(wishart(), HMC(0.01, 1), 1000);
        #Turing.setadbackend(:reversediff)
        #sample(wishart(), HMC(0.01, 1), 1000);
        Turing.setadbackend(:zygote)
        sample(wishart(), HMC(0.01, 1), 1000);

        @model invwishart() = begin
            theta ~ InverseWishart(4, Matrix{Float64}(I, 4, 4))
        end
        Turing.setadbackend(:tracker)
        sample(invwishart(), HMC(0.01, 1), 1000);
        #Turing.setadbackend(:reversediff)
        #sample(invwishart(), HMC(0.01, 1), 1000);
        Turing.setadbackend(:zygote)
        sample(invwishart(), HMC(0.01, 1), 1000);
    end
    @testset "Hessian test" begin
        @model function tst(x, ::Type{TV}=Vector{Float64}) where {TV}
            params = TV(undef, 2)
            @. params ~ Normal(0, 1)
        
            x ~ MvNormal(params, 1)
        end
        
        function make_logjoint(model::DynamicPPL.Model, ctx::DynamicPPL.AbstractContext)
            # setup
            varinfo_init = Turing.VarInfo(model)
            spl = DynamicPPL.SampleFromPrior()    
            DynamicPPL.link!(varinfo_init, spl)
        
            function logπ(z; unlinked = false)
                varinfo = DynamicPPL.VarInfo(varinfo_init, spl, z)
        
                unlinked && DynamicPPL.invlink!(varinfo_init, spl)
                model(varinfo, spl, ctx)
                unlinked && DynamicPPL.link!(varinfo_init, spl)
        
                return -DynamicPPL.getlogp(varinfo)
            end
        
            return logπ
        end
        
        data = [0.5, -0.5]
        model = tst(data)
        
        likelihood = make_logjoint(model, DynamicPPL.LikelihoodContext())
        target(x) = likelihood(x, unlinked=true)
        
        H_f = ForwardDiff.hessian(target, zeros(2))
        H_r = ReverseDiff.hessian(target, zeros(2))
        @test H_f == [1.0 0.0; 0.0 1.0]
        @test H_f == H_r
    end
end
