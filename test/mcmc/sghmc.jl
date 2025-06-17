module SGHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using DynamicPPL.TestUtils.AD: run_ad
using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL: DynamicPPL
using Distributions: sample
import ForwardDiff
using LinearAlgebra: dot
using Random: Xoshiro
using StableRNGs: StableRNG
using Test: @test, @testset
using Turing

@testset "SGHMC + SGLD: InferenceAlgorithm interface" begin
    algs = [
        SGHMC(; learning_rate=0.01, momentum_decay=0.1),
        SGLD(; stepsize=PolynomialStepsize(0.25)),
    ]

    @testset "get_adtype" begin
        # Default
        for alg in algs
            @test Turing.Inference.get_adtype(alg) == Turing.DEFAULT_ADTYPE
        end
        # Manual
        for adtype in (AutoReverseDiff(), AutoMooncake(; config=nothing))
            alg1 = SGHMC(; learning_rate=0.01, momentum_decay=0.1, adtype=adtype)
            alg2 = SGLD(; stepsize=PolynomialStepsize(0.25), adtype=adtype)
            @test Turing.Inference.get_adtype(alg1) == adtype
            @test Turing.Inference.get_adtype(alg2) == adtype
        end
    end

    @testset "requires_unconstrained_space" begin
        # Hamiltonian samplers always need it
        for alg in algs
            @test Turing.Inference.requires_unconstrained_space(alg)
        end
    end

    @testset "update_sample_kwargs" begin
        # These don't update kwargs
        for alg in algs
            kwargs = (a=1, b=2)
            @test Turing.Inference.update_sample_kwargs(alg, 1000, kwargs) == kwargs
        end
    end
end

@testset verbose = true "SGHMC + SGLD: sample() interface" begin
    @model function demo_normal(x)
        a ~ Normal()
        return x ~ Normal(a)
    end
    model = demo_normal(2.0)
    # note: passing LDF to a Hamiltonian sampler requires explicit adtype
    ldf = LogDensityFunction(model; adtype=AutoForwardDiff())
    sampling_objects = Dict("DynamicPPL.Model" => model, "LogDensityFunction" => ldf)
    algs = [
        SGHMC(; learning_rate=0.01, momentum_decay=0.1),
        SGLD(; stepsize=PolynomialStepsize(0.25)),
    ]
    seed = 468
    @testset "sampling with $name" for (name, model_or_ldf) in sampling_objects
        @testset "$alg" for alg in algs
            # check sampling works without rng
            @test sample(model_or_ldf, alg, 5) isa Chains
            # check reproducibility with rng
            chn1 = sample(Xoshiro(seed), model_or_ldf, alg, 5)
            chn2 = sample(Xoshiro(seed), model_or_ldf, alg, 5)
            @test mean(chn1[:a]) == mean(chn2[:a])
        end
    end
end

@testset verbose = true "Testing sghmc.jl" begin
    @testset "sghmc constructor" begin
        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
    end

    @testset "sghmc inference" begin
        rng = StableRNG(123)
        alg = SGHMC(; learning_rate=0.02, momentum_decay=0.5)
        chain = sample(rng, gdemo_default, alg, 10_000)
        check_gdemo(chain; atol=0.1)
    end
end

@testset "Testing sgld.jl" begin
    @testset "sgld constructor" begin
        alg = SGLD(; stepsize=PolynomialStepsize(0.25))
        @test alg isa SGLD
    end
    @testset "sgld inference" begin
        rng = StableRNG(1)

        chain = sample(rng, gdemo_default, SGLD(; stepsize=PolynomialStepsize(0.5)), 20_000)
        check_gdemo(chain; atol=0.2)

        # Weight samples by step sizes (cf section 4.2 in the paper by Welling and Teh)
        v = get(chain, [:SGLD_stepsize, :s, :m])
        s_weighted = dot(v.SGLD_stepsize, v.s) / sum(v.SGLD_stepsize)
        m_weighted = dot(v.SGLD_stepsize, v.m) / sum(v.SGLD_stepsize)
        @test s_weighted ≈ 49 / 24 atol = 0.2
        @test m_weighted ≈ 7 / 6 atol = 0.2
    end
end

end
