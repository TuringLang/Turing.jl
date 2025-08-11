module SGHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using DynamicPPL.TestUtils.AD: run_ad
using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL: DynamicPPL
using Distributions: sample
import ForwardDiff
using LinearAlgebra: dot
import ReverseDiff
using StableRNGs: StableRNG
using Test: @test, @testset
using Turing

@testset verbose = true "Testing sghmc.jl" begin
    @testset "sghmc constructor" begin
        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
        sampler = DynamicPPL.Sampler(alg)
        @test sampler isa DynamicPPL.Sampler{<:SGHMC}

        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
        sampler = DynamicPPL.Sampler(alg)
        @test sampler isa DynamicPPL.Sampler{<:SGHMC}
    end

    @testset "sghmc inference" begin
        rng = StableRNG(123)
        alg = SGHMC(; learning_rate=0.02, momentum_decay=0.5)
        chain = sample(rng, gdemo_default, alg, 10_000)
        check_gdemo(chain; atol=0.1)
    end

    @testset "chain log-density metadata" begin
        @model function f()
            x ~ LogNormal()
            return 1.0 ~ Normal(x)
        end
        N = 100
        chn = sample(f(), SGHMC(; learning_rate=0.02, momentum_decay=0.5), N)
        @test chn[:logprior] ≈ logpdf.(LogNormal(), chn[:x])
        @test chn[:loglikelihood] ≈ logpdf.(Normal.(chn[:x]), 1.0)
        @test chn[:lp] ≈ chn[:logprior] + chn[:loglikelihood]
    end
end

@testset "Testing sgld.jl" begin
    @testset "sgld constructor" begin
        alg = SGLD(; stepsize=PolynomialStepsize(0.25))
        @test alg isa SGLD
        sampler = DynamicPPL.Sampler(alg)
        @test sampler isa DynamicPPL.Sampler{<:SGLD}

        alg = SGLD(; stepsize=PolynomialStepsize(0.25))
        @test alg isa SGLD
        sampler = DynamicPPL.Sampler(alg)
        @test sampler isa DynamicPPL.Sampler{<:SGLD}
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

    @testset "chain log-density metadata" begin
        @model function f()
            x ~ LogNormal()
            return 1.0 ~ Normal(x)
        end
        N = 100
        chn = sample(f(), SGLD(; stepsize=PolynomialStepsize(0.25)), N)
        @test chn[:logprior] ≈ logpdf.(LogNormal(), chn[:x])
        @test chn[:loglikelihood] ≈ logpdf.(Normal.(chn[:x]), 1.0)
        @test chn[:lp] ≈ chn[:logprior] + chn[:loglikelihood]
    end
end

end
