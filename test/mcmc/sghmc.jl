module SGHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
import ..ADUtils
using DynamicPPL.TestUtils.AD: run_ad
using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL: DynamicPPL
using Distributions: sample
import ForwardDiff
using LinearAlgebra: dot
import ReverseDiff
using StableRNGs: StableRNG
import Mooncake
using Test: @test, @testset
using Turing

@testset "AD with SGHMC / SGLD" begin
    @testset "adtype=$adtype" for adtype in ADUtils.adbackends
        @testset "alg=$alg" for alg in [
            SGHMC(; learning_rate=0.02, momentum_decay=0.5, adtype=adtype),
            SGLD(; stepsize=PolynomialStepsize(0.25), adtype=adtype),
        ]
            @testset "model=$(model.f)" for model in DEMO_MODELS
                rng = StableRNG(123)
                ctx = DynamicPPL.SamplingContext(rng, DynamicPPL.Sampler(alg))
                @test run_ad(model, adtype; context=ctx, test=true, benchmark=false) isa Any
            end
        end
    end
end

@testset "Testing sghmc.jl" begin
    @testset "sghmc constructor" begin
        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1, adtype=Turing.DEFAULT_ADTYPE)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}

        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1, adtype=Turing.DEFAULT_ADTYPE)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}
    end
    @testset "sghmc inference" begin
        rng = StableRNG(123)

        alg = SGHMC(; learning_rate=0.02, momentum_decay=0.5, adtype=Turing.DEFAULT_ADTYPE)
        chain = sample(rng, gdemo_default, alg, 10_000)
        check_gdemo(chain; atol=0.1)
    end
end

@testset "Testing sgld.jl" begin
    @testset "sgld constructor" begin
        alg = SGLD(; stepsize=PolynomialStepsize(0.25), adtype=Turing.DEFAULT_ADTYPE)
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}

        alg = SGLD(; stepsize=PolynomialStepsize(0.25), adtype=Turing.DEFAULT_ADTYPE)
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}
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
