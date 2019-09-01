using ForwardDiff, Distributions, FiniteDifferences, Tracker, Random, LinearAlgebra, PDMats
using Turing: Turing, gradient_logp_reverse, invlink, link, SampleFromPrior
using Turing.Core.RandomVariables: getval
using Turing.Core: TuringMvNormal, TuringDiagNormal
using ForwardDiff: Dual
using StatsFuns: binomlogpdf, logsumexp
using Test, LinearAlgebra
const FDM = FiniteDifferences
using Combinatorics

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

_to_cov(B) = B * B' + Matrix(I, size(B)...)
struct ADTestFunction
    name::String
    f::Function
    x::Vector
end
struct DistSpec{Tθ<:Tuple, Tx}
    name::Union{Symbol, Expr}
    θ::Tθ
    x::Tx
end

@testset "ad.jl" begin
    @turing_testset "AD compatibility" begin        
        vectorize(v::Number) = [v]
        vectorize(v) = vec(v)
        pack(vals...) = reduce(vcat, vectorize.(vals))
        function unpack(x, vals...)
            unpacked = []
            i = 1
            for v in vals
                if v isa Number
                    push!(unpacked, x[i])
                    i += 1
                elseif v isa Vector
                    push!(unpacked, x[i:i+length(v)-1])
                    i += length(v)
                elseif v isa Matrix
                    push!(unpacked, reshape(x[i:(i+length(v)-1)], size(v)))
                    i += length(v)
                else
                    throw("Unsupported argument")
                end
            end
            @assert i == length(x) + 1
            return (unpacked...,)
        end
        function get_function(dist::DistSpec, inds, val)
            syms = []
            args = []
            for (i, a) in enumerate(dist.θ)
                if i in inds
                    sym = gensym()
                    push!(syms, sym)
                    push!(args, sym)
                else
                    push!(args, a)
                end
            end
            if val
                sym = gensym()
                push!(syms, sym)
                expr = :(($(syms...),) -> logpdf($(dist.name)($(args...)), $(sym)))
                if length(inds) == 0
                    f = x -> Base.invokelatest(eval(expr), unpack(x, dist.x)...)
                    return ADTestFunction(string(expr), f, pack(dist.x))
                else
                    f = x -> Base.invokelatest(eval(expr), unpack(x, dist.θ[inds]..., dist.x)...)
                    return ADTestFunction(string(expr), f, pack(dist.θ[inds]..., dist.x))
                end
            else
                @assert length(inds) > 0
                expr = :(($(syms...),) -> logpdf($(dist.name)($(args...)), $(dist.x)))
                f = x -> Base.invokelatest(eval(expr), unpack(x, dist.θ[inds]...)...)
                return ADTestFunction(string(expr), f, pack(dist.θ[inds]...))
            end
        end
        function get_all_functions(dist::DistSpec, continuous=false)
            fs = []
            if length(dist.θ) == 0
                push!(fs, get_function(dist, (), true))
            else
                for inds in combinations(1:length(dist.θ))
                    push!(fs, get_function(dist, inds, false))
                    if continuous
                        push!(fs, get_function(dist, inds, true))
                    end
                end
            end
            return fs
        end
        dim = 3
        mean = zeros(dim)
        cov_mat = Matrix{Float64}(I, dim, dim)
        cov_vec = ones(dim)
        cov_num = 1.0
        norm_val = ones(dim)
        alpha = ones(4)
        dir_val = fill(0.25, 4)

        uni_disc_dists = [
            DistSpec(:Bernoulli, (0.45,), 1),
            DistSpec(:Bernoulli, (0.45,), 0),
            DistSpec(:((a, b) -> BetaBinomial(10, a, b)), (2, 1), 5),
            DistSpec(:(p -> Binomial(10, p)), (0.5,), 5),
            DistSpec(:Categorical, ([0.45, 0.55],), 1),
            DistSpec(:Geometric, (0.45,), 3),
            DistSpec(:NegativeBinomial, (3.5, 0.5), 1),
            DistSpec(:Poisson, (0.5,), 1),
            DistSpec(:Skellam, (1.0, 2.0), -2),
        ]
        uni_cont_dists = [
            DistSpec(:Arcsine, (), 0.5),
            DistSpec(:Arcsine, (1,), 0.5),
            DistSpec(:Arcsine, (0, 2), 0.5),
            DistSpec(:Beta, (), 0.5),
            DistSpec(:Beta, (1,), 0.5),
            DistSpec(:Beta, (1, 2), 0.5),
            DistSpec(:BetaPrime, (), 0.5),
            DistSpec(:BetaPrime, (1,), 0.5),
            DistSpec(:BetaPrime, (1, 2), 0.5),
            DistSpec(:Biweight, (), 0.5),
            DistSpec(:Biweight, (1,), 0.5),
            DistSpec(:Biweight, (1, 2), 0.5),
            DistSpec(:Cauchy, (), 0.5),
            DistSpec(:Cauchy, (1,), 0.5),
            DistSpec(:Cauchy, (1, 2), 0.5),
            DistSpec(:Chernoff, (), 0.5),
            DistSpec(:Chi, (1,), 0.5),
            DistSpec(:Chisq, (1,), 0.5),
            DistSpec(:Cosine, (1, 1), 0.5),
            DistSpec(:Epanechnikov, (1, 1), 0.5),
            DistSpec(:((s)->Erlang(1, s)), (1,), 0.5), # First arg is integer
            DistSpec(:Exponential, (1,), 0.5),
            DistSpec(:FDist, (1, 1), 0.5),
            DistSpec(:Frechet, (), 0.5),
            DistSpec(:Frechet, (1,), 0.5),
            DistSpec(:Frechet, (1, 2), 0.5),
            DistSpec(:Gamma, (), 0.5),
            DistSpec(:Gamma, (1,), 0.5),
            DistSpec(:Gamma, (1, 2), 0.5),
            DistSpec(:GeneralizedExtremeValue, (1.0, 1.0, 1.0), 0.5),
            DistSpec(:GeneralizedPareto, (), 0.5),
            DistSpec(:GeneralizedPareto, (1.0, 2.0), 0.5),
            DistSpec(:GeneralizedPareto, (0.0, 2.0, 3.0), 0.5),
            DistSpec(:Gumbel, (), 0.5),
            DistSpec(:Gumbel, (1,), 0.5),
            DistSpec(:Gumbel, (1, 2), 0.5),
            DistSpec(:InverseGamma, (), 0.5),
            DistSpec(:InverseGamma, (1.0,), 0.5),
            DistSpec(:InverseGamma, (1.0, 2.0), 0.5),
            DistSpec(:InverseGaussian, (), 0.5),
            DistSpec(:InverseGaussian, (1,), 0.5),
            DistSpec(:InverseGaussian, (1, 2), 0.5),
            DistSpec(:Kolmogorov, (), 0.5),
            DistSpec(:Laplace, (), 0.5),
            DistSpec(:Laplace, (1,), 0.5),
            DistSpec(:Laplace, (1, 2), 0.5),
            DistSpec(:Levy, (), 0.5),
            DistSpec(:Levy, (0.0,), 0.5),
            DistSpec(:Levy, (0.0, 2.0), 0.5),
            DistSpec(:((a, b) -> LocationScale(a, b, Normal())), (1.0, 2.0), 0.5),
            DistSpec(:Logistic, (), 0.5),
            DistSpec(:Logistic, (1,), 0.5),
            DistSpec(:Logistic, (1, 2), 0.5),
            DistSpec(:LogitNormal, (), 0.5),
            DistSpec(:LogitNormal, (1,), 0.5),
            DistSpec(:LogitNormal, (1, 2), 0.5),
            DistSpec(:LogNormal, (), 0.5),
            DistSpec(:LogNormal, (1,), 0.5),
            DistSpec(:LogNormal, (1, 2), 0.5),
            DistSpec(:Normal, (), 0.5),
            DistSpec(:Normal, (1.0,), 0.5),
            DistSpec(:Normal, (1.0, 2.0), 0.5),
            DistSpec(:NormalCanon, (1.0, 2.0), 0.5),
            DistSpec(:NormalInverseGaussian, (1.0, 2.0, 1.0, 1.0), 0.5),
            DistSpec(:Pareto, (), 1.5),
            DistSpec(:Pareto, (1,), 1.5),
            DistSpec(:Pareto, (1, 1), 1.5),
            DistSpec(:PGeneralizedGaussian, (), 0.5),
            DistSpec(:PGeneralizedGaussian, (1, 1, 1), 0.5),
            DistSpec(:Rayleigh, (), 0.5),
            DistSpec(:Rayleigh, (1,), 0.5),
            DistSpec(:SymTriangularDist, (), 0.5),
            DistSpec(:SymTriangularDist, (1,), 0.5),
            DistSpec(:SymTriangularDist, (1, 2), 0.5),
            DistSpec(:TDist, (1,), 0.5),
            DistSpec(:TriangularDist, (1, 2), 1.5),
            DistSpec(:TriangularDist, (1, 3, 2), 1.5),
            DistSpec(:Triweight, (1, 1), 1),
            DistSpec(:Uniform, (0, 1), 0.5),
            DistSpec(:VonMises, (), 1),
            DistSpec(:Weibull, (), 1),
            DistSpec(:Weibull, (1,), 1),
            DistSpec(:Weibull, (1, 1), 1),
        ]
        mult_disc_dists = [
        ]
        mult_cont_dists = [
            DistSpec(:MvNormal, (mean, cov_mat), norm_val),
            DistSpec(:MvNormal, (mean, cov_vec), norm_val),
            DistSpec(:MvNormal, (mean, cov_num), norm_val),
            DistSpec(:MvNormal, (cov_mat,), norm_val),
            DistSpec(:MvNormal, (cov_vec,), norm_val),
            DistSpec(:(cov_num -> MvNormal(dim, cov_num)), (cov_num,), norm_val),
            DistSpec(:MvLogNormal, (mean, cov_mat), norm_val),
            DistSpec(:MvLogNormal, (mean, cov_vec), norm_val),
            DistSpec(:MvLogNormal, (mean, cov_num), norm_val),
            DistSpec(:MvLogNormal, (cov_mat,), norm_val),
            DistSpec(:MvLogNormal, (cov_vec,), norm_val),
            DistSpec(:(cov_num -> MvLogNormal(dim, cov_num)), (cov_num,), norm_val),
        ]

        broken_uni_disc_dists = [
            # Dispatch error
            DistSpec(:PoissonBinomial, ([0.5, 0.5],), 3),
        ]
        broken_uni_cont_dists = [
            # Broken in Distributions even without autodiff
            DistSpec(:(()->KSDist(1)), (), 0.5), 
            DistSpec(:(()->KSOneSided(1)), (), 0.5), 
            DistSpec(:StudentizedRange, (1.0, 2.0), 0.5),
            # Dispatch error
            DistSpec(:NoncentralBeta, (1.0, 2.0, 1.0), 0.5), 
            DistSpec(:NoncentralChisq, (1.0, 2.0), 0.5),
            DistSpec(:NoncentralF, (1, 2, 1), 0.5),
            DistSpec(:NoncentralT, (1, 2), 0.5),
            DistSpec(:((mu, sigma, l, u) -> Truncated(Normal(mu, sigma), l, u)), (0.0, 1.0, 1.0, 2.0), 1.5),
            # Possibly Tracker error
            DistSpec(:Uniform, (), 0.5),
            DistSpec(:Semicircle, (1.0,), 0.5),
            # Stackoverflow
            DistSpec(:VonMises, (1.0,), 1.0),
            DistSpec(:VonMises, (1, 1), 1),
        ]
        broken_mult_disc_dists = [
            # Dispatch error
            DistSpec(:((p) -> Multinomial(4, p)), (fill(0.25, 4),), 1),
        ]
        broken_mult_cont_dists = [
            # Dispatch error
            DistSpec(:MvNormalCanon, (mean, cov_mat), norm_val),
            DistSpec(:MvNormalCanon, (mean, cov_vec), norm_val),
            DistSpec(:MvNormalCanon, (mean, cov_num), norm_val),
            DistSpec(:MvNormalCanon, (cov_mat,), norm_val),
            DistSpec(:MvNormalCanon, (cov_vec,), norm_val),
            DistSpec(:(cov_num -> MvNormalCanon(dim, cov_num)), (cov_num,), norm_val),
            DistSpec(:Dirichlet, (alpha,), dir_val),
            # Test failure
            DistSpec(:(() -> Product(Normal.(randn(dim), 1))), (), norm_val),
        ]

        for d in uni_disc_dists
            for testf in get_all_functions(d, false)
                test_ad(testf.f, testf.x)
            end
        end
        for d in uni_cont_dists
            for testf in get_all_functions(d, true)
                test_ad(testf.f, testf.x)
            end
        end
        for d in mult_disc_dists
            for testf in get_all_functions(d, false)
                test_ad(testf.f, testf.x)
            end
        end
        for d in mult_cont_dists
            for testf in get_all_functions(d, true)
                test_ad(testf.f, testf.x)
            end
        end
    end
    @turing_testset "adr" begin
        ad_test_f = gdemo_default
        vi = Turing.VarInfo()
        ad_test_f(vi, SampleFromPrior())
        svn = vi.vns[1]
        mvn = vi.vns[2]
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
    @turing_testset "Tracker + logdet" begin
        rng, N = MersenneTwister(123456), 7
        ȳ, B = randn(rng), randn(rng, N, N)
        test_tracker_ad(B->logdet(cholesky(_to_cov(B))), ȳ, B; rtol=1e-8, atol=1e-8)
    end
    @turing_testset "Tracker + fill" begin
        rng = MersenneTwister(123456)
        test_tracker_ad(x->fill(x, 7), randn(rng, 7), randn(rng))
        test_tracker_ad(x->fill(x, 7, 11), randn(rng, 7, 11), randn(rng))
        test_tracker_ad(x->fill(x, 7, 11, 13), rand(rng, 7, 11, 13), randn(rng))
    end
    @turing_testset "Tracker + MvNormal" begin
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
    @turing_testset "Tracker + Diagonal Normal" begin
        rng, N = MersenneTwister(123456), 11
        m, σ = randn(rng, N), exp.(0.1 .* randn(rng, N)) .+ 1

        d = TuringDiagNormal(m, σ)
        x = rand(d)

        # Check that the logpdf agrees with MvNormal.
        d_ref = MvNormal(m, σ)
        @test logpdf(d, x) ≈ logpdf(d_ref, x)

        test_tracker_ad((m, σ, x)->logpdf(MvNormal(m, σ), x), randn(rng), m, σ, x)
    end
    @turing_testset "Tracker + MvNormal Interface" begin
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
    @testset "Simplex Tracker AD" begin
        @model dir() = begin
            theta ~ Dirichlet(1 ./ fill(4, 4))
        end
        Turing.setadbackend(:reverse_diff)
        sample(dir(), HMC(0.01, 1), 1000);
    end
    @testset "PDMatDistribution Tracker AD" begin
        @model wishart() = begin
            theta ~ Wishart(4, Matrix{Float64}(I, 4, 4))
        end
        Turing.setadbackend(:reverse_diff)
        sample(wishart(), HMC(0.01, 1), 1000);

        @model invwishart() = begin
            theta ~ InverseWishart(4, Matrix{Float64}(I, 4, 4))
        end
        Turing.setadbackend(:reverse_diff)
        sample(invwishart(), HMC(0.01, 1), 1000);
    end
end
