
module AdvancedVITests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo

using AbstractMCMC: AbstractMCMC
using AdvancedVI
using Bijectors: Bijectors
using Distributions: Dirichlet, Normal
using DynamicPPL: DynamicPPL
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using MCMCChains: Chains
using Random
using ReverseDiff
using StableRNGs: StableRNG
using Test: @test, @testset, @test_throws
using Turing
using Turing.Variational

begin
    adtype = AutoReverseDiff()
    operator = AdvancedVI.ClipScale()

    @testset "q initialization" begin
        m = gdemo_default
        l = LogDensityFunction(m, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll())
        d = LogDensityProblems.dimension(l)

        for q in [q_meanfield_gaussian(l), q_fullrank_gaussian(l)]
            rand(q)
        end

        μ = ones(d)
        q = q_meanfield_gaussian(l; location=μ)
        @test mean(q) ≈ μ

        q = q_fullrank_gaussian(l; location=μ)
        @test mean(q) ≈ μ

        L = Diagonal(fill(0.1, d))
        q = q_meanfield_gaussian(l; scale=L)
        @test cov(q) ≈ L * L

        L = LowerTriangular(tril(0.01 * ones(d, d) + I))
        q = q_fullrank_gaussian(l; scale=L)
        @test cov(q) ≈ L * L'
    end

    @testset "default interface" begin
        for q0 in [q_meanfield_gaussian, q_fullrank_gaussian]
            result = vi(gdemo_default, q0, 100; show_progress=Turing.PROGRESS[], adtype)
            @test result isa Turing.Variational.VIResult
            @test rand(result) isa DynamicPPL.VarNamedTuple
            @test rand(result, 2) isa Vector{<:DynamicPPL.VarNamedTuple}
            @test size(rand(result, 2)) == (2,)
            @test rand(result, 5, 2) isa Matrix{<:DynamicPPL.VarNamedTuple}
            @test size(rand(result, 5, 2)) == (5, 2)
        end
    end

    @testset "custom algorithm $name" for (name, algorithm) in [
        ("KLMinRepGradProxDescent", KLMinRepGradProxDescent(adtype; n_samples=10)),
        ("KLMinRepGradDescent", KLMinRepGradDescent(adtype; operator, n_samples=10)),
        ("KLMinNaturalGradDescent", KLMinNaturalGradDescent(; stepsize=1e-2, n_samples=10)),
        (
            "KLMinSqrtNaturalGradDescent",
            KLMinSqrtNaturalGradDescent(; stepsize=1e-2, n_samples=10),
        ),
        ("KLMinWassFwdBwd", KLMinWassFwdBwd(; stepsize=1e-2, n_samples=10)),
        ("FisherMinBatchMatch", FisherMinBatchMatch()),
    ]
        T = 1000
        result = vi(
            gdemo_default,
            q_fullrank_gaussian,
            T;
            algorithm,
            show_progress=Turing.PROGRESS[],
        )
        c2 = rand(result, 10)
        @test c2 isa Vector{<:DynamicPPL.VarNamedTuple}
    end

    @testset "inference $name" for (name, algorithm) in [
        ("KLMinRepGradProxDescent", KLMinRepGradProxDescent(adtype; n_samples=10)),
        ("KLMinRepGradDescent", KLMinRepGradDescent(adtype; operator, n_samples=100)),
        (
            "KLMinNaturalGradDescent",
            KLMinNaturalGradDescent(; stepsize=1e-2, n_samples=100),
        ),
        (
            "KLMinSqrtNaturalGradDescent",
            KLMinSqrtNaturalGradDescent(; stepsize=1e-2, n_samples=100),
        ),
        ("KLMinWassFwdBwd", KLMinWassFwdBwd(; stepsize=1e-2, n_samples=10)),
        ("FisherMinBatchMatch", FisherMinBatchMatch()),
    ]
        rng = StableRNG(468)

        T = 1000
        result = vi(
            rng,
            gdemo_default,
            q_fullrank_gaussian,
            T;
            algorithm,
            show_progress=Turing.PROGRESS[],
        )

        N = 1000
        samples = rand(rng, result, N)
        chn = AbstractMCMC.from_samples(MCMCChains.Chains, hcat(samples))

        check_gdemo(chn; atol=0.5)
    end

    # regression test for https://github.com/TuringLang/Turing.jl/issues/2065
    # and https://github.com/TuringLang/Turing.jl/issues/2160
    @testset "simplex bijector" begin
        rng = StableRNG(0x517e1d9bf89bf94f)

        @model function dirichlet()
            x ~ Dirichlet([1.0, 1.0])
            return x
        end
        m = dirichlet()
        result = vi(rng, m, q_meanfield_gaussian, 1000)
        samples = rand(rng, result, 1000)
        @test mean(s -> s[@varname(x)], samples) ≈ [0.5, 0.5] atol = 0.1
    end

    # Ref: https://github.com/TuringLang/Turing.jl/issues/2205
    @testset "with `condition` (issue #2205)" begin
        rng = StableRNG(0x517e1d9bf89bf94f)

        @model function demo_issue2205()
            x ~ Normal()
            return y ~ Normal(x, 1)
        end

        model = demo_issue2205() | (y=1.0,)
        result = vi(rng, model, q_meanfield_gaussian, 1000)
        # True mean.
        mean_true = 1 / 2
        var_true = 1 / 2
        # Check the mean and variance of the posterior.
        samples = rand(rng, result, 1000)
        xs = [s[@varname(x)] for s in samples]
        @test mean(xs) ≈ mean_true atol = 0.2
        @test var(xs) ≈ var_true atol = 0.2
    end

    @testset "fix_transforms" begin
        struct MyNormal <: ContinuousUnivariateDistribution end
        Distributions.logpdf(::MyNormal, x) = logpdf(Normal(), x)
        Distributions.rand(rng::Random.AbstractRNG, ::MyNormal) = rand(rng, Normal())
        counter = Ref(0)
        struct VectAndIncrement end
        (::VectAndIncrement)(x) = [x]
        Bijectors.with_logabsdet_jacobian(::VectAndIncrement, x) = [x], 0.0
        Bijectors.inverse(::VectAndIncrement) = OnlyAndIncrement()
        struct OnlyAndIncrement end
        (::OnlyAndIncrement)(x) = x[]
        Bijectors.with_logabsdet_jacobian(::OnlyAndIncrement, x) = x[], 0.0
        Bijectors.inverse(::OnlyAndIncrement) = VectAndIncrement()
        function Bijectors.VectorBijectors.to_linked_vec(::MyNormal)
            counter[] += 1
            return VectAndIncrement()
        end
        function Bijectors.VectorBijectors.from_linked_vec(::MyNormal)
            counter[] += 1
            return OnlyAndIncrement()
        end

        @model f() = x ~ MyNormal()
        model = f()

        counter[] = 0
        vi(model, q_meanfield_gaussian, 100)
        @test counter[] > 100

        counter[] = 0
        vi(model, q_meanfield_gaussian, 100; fix_transforms=true)
        @test counter[] < 100
    end
end

end
