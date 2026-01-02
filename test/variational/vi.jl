module AdvancedVITests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo

using AdvancedVI
using Bijectors: Bijectors
using FlexiChains: FlexiChain, Parameter
using LinearAlgebra
using Random
import ReverseDiff
using StableRNGs: StableRNG
using Test: @test, @testset, @test_throws
using Turing
using Turing.Variational

@testset verbose = true "variational/vi.jl" begin
    adtype = AutoReverseDiff()
    operator = AdvancedVI.ClipScale()

    @testset "q initialization" begin
        m = gdemo_default
        d = length(Turing.DynamicPPL.VarInfo(m)[:])
        for q in [q_meanfield_gaussian(m), q_fullrank_gaussian(m)]
            rand(q)
        end

        μ = ones(d)
        q = q_meanfield_gaussian(m; location=μ)
        @assert mean(q.dist) ≈ μ

        q = q_fullrank_gaussian(m; location=μ)
        @assert mean(q.dist) ≈ μ

        L = Diagonal(fill(0.1, d))
        q = q_meanfield_gaussian(m; scale=L)
        @assert cov(q.dist) ≈ L * L

        L = LowerTriangular(tril(0.01 * ones(d, d) + I))
        q = q_fullrank_gaussian(m; scale=L)
        @assert cov(q.dist) ≈ L * L'
    end

    @testset "default interface" begin
        for q0 in [q_meanfield_gaussian(gdemo_default), q_fullrank_gaussian(gdemo_default)]
            q, _, _ = vi(gdemo_default, q0, 100; show_progress=Turing.PROGRESS[], adtype)
            c1 = rand(q, 10)
        end
        @test_throws "unconstrained" begin
            q, _, _ = vi(gdemo_default, Normal(), 1; adtype)
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
        q, _, _ = vi(
            gdemo_default,
            q_fullrank_gaussian(gdemo_default),
            T;
            algorithm,
            show_progress=Turing.PROGRESS[],
        )
        N = 1000
        c2 = rand(q, N)
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
        rng = StableRNG(0x517e1d9bf89bf94f)

        T = 1000
        q, _, _ = vi(
            rng,
            gdemo_default,
            q_fullrank_gaussian(gdemo_default),
            T;
            algorithm,
            show_progress=Turing.PROGRESS[],
        )

        N = 1000
        # 2 * N matrix
        samples = rand(rng, q, N)
        samples_dict = Dict(
            Parameter(@varname(s)) => samples[1, :], Parameter(@varname(m)) => samples[2, :]
        )
        chn = FlexiChain{VarName}(N, 1, samples_dict)

        check_gdemo(chn; atol=0.5)
    end

    # regression test for:
    # https://github.com/TuringLang/Turing.jl/issues/2065
    @testset "simplex bijector" begin
        rng = StableRNG(0x517e1d9bf89bf94f)

        @model function dirichlet()
            x ~ Dirichlet([1.0, 1.0])
            return x
        end

        m = dirichlet()
        b = Bijectors.bijector(m)
        x0 = m()
        z0 = b(x0)
        @test size(z0) == (1,)
        x0_inv = Bijectors.inverse(b)(z0)
        @test size(x0_inv) == size(x0)
        @test all(x0 .≈ x0_inv)

        # And regression for https://github.com/TuringLang/Turing.jl/issues/2160.
        q, _, _ = vi(rng, m, q_meanfield_gaussian(m), 1000)
        x = rand(rng, q, 1000)
        @test mean(eachcol(x)) ≈ [0.5, 0.5] atol = 0.1
    end

    # Ref: https://github.com/TuringLang/Turing.jl/issues/2205
    @testset "with `condition` (issue #2205)" begin
        rng = StableRNG(0x517e1d9bf89bf94f)

        @model function demo_issue2205()
            x ~ Normal()
            return y ~ Normal(x, 1)
        end

        model = demo_issue2205() | (y=1.0,)
        q, _, _ = vi(rng, model, q_meanfield_gaussian(model), 1000)
        # True mean.
        mean_true = 1 / 2
        var_true = 1 / 2
        # Check the mean and variance of the posterior.
        samples = rand(rng, q, 1000)
        mean_est = mean(samples)
        var_est = var(samples)
        @test mean_est ≈ mean_true atol = 0.2
        @test var_est ≈ var_true atol = 0.2
    end
end

end
