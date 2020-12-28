using Turing
using LinearAlgebra
using Random
using Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "sghmc.jl" begin
    @numerical_testset "sghmc inference" begin
        Random.seed!(125)

        alg = SGHMC(; learning_rate=0.02, momentum_decay=0.5)
        chain = sample(gdemo_default, alg, 10_000)
        check_gdemo(chain, atol = 0.1)
    end
    @turing_testset "sghmc constructor" begin
        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}

        alg = SGHMC(:m; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}

        alg = SGHMC(:s; learning_rate=0.01, momentum_decay=0.1)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}
    end
end

@testset "sgld.jl" begin
    @turing_testset "sgld constructor" begin
        alg = SGLD(; stepsize = PolynomialStepsize(0.25))
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}

        alg = SGLD(:m; stepsize = PolynomialStepsize(0.25))
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}

        alg = SGLD(:s; stepsize = PolynomialStepsize(0.25))
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}
    end
    @numerical_testset "sgld inference" begin
        Random.seed!(125)

        chain = sample(gdemo_default, SGLD(; stepsize = PolynomialStepsize(0.5)), 10_000)
        check_gdemo(chain, atol = 0.2)

        # Weight samples by step sizes (cf section 4.2 in the paper by Welling and Teh)
        v = get(chain, [:SGLD_stepsize, :s, :m])
        s_weighted = dot(v.SGLD_stepsize, v.s) / sum(v.SGLD_stepsize)
        m_weighted = dot(v.SGLD_stepsize, v.m) / sum(v.SGLD_stepsize)
        @test s_weighted ≈ 49/24 atol=0.2
        @test m_weighted ≈ 7/6 atol=0.2
    end
end
