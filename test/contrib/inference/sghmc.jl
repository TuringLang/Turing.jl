using Turing, Random, Test
using Turing: Sampler

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "sghmc.jl" begin
    @numerical_testset "sghmc inference" begin
        Random.seed!(125)

        alg = SGHMC(10000, 0.02, 0.5)
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps=0.1)
    end
    @turing_testset "sghmc constructor" begin
        alg = SGHMC(1000, 0.01, 0.1)
        sampler = Sampler(alg)
        @test isa(alg, SGHMC)
        @test isa(sampler, Sampler{<:Turing.SGHMC})

        alg = SGHMC(200, 0.01, 0.1, :m)
        sampler = Sampler(alg)
        @test isa(alg, SGHMC)
        @test isa(sampler, Sampler{<:Turing.SGHMC})

        alg = SGHMC(1000, 0.01, 0.1, :s)
        sampler = Sampler(alg)
        @test isa(alg, SGHMC)
        @test isa(sampler, Sampler{<:Turing.SGHMC})
    end
end

@testset "sgld.jl" begin
    @turing_testset "sgld constructor" begin
        alg = SGLD(1000, 0.25)
        sampler = Sampler(alg)
        @test isa(alg, SGLD)
        @test isa(sampler, Sampler{<:Turing.SGLD})

        alg = SGLD(200, 0.25, :m)
        sampler = Sampler(alg)
        @test isa(alg, SGLD)
        @test isa(sampler, Sampler{<:Turing.SGLD})

        alg = SGLD(1000, 0.25, :s)
        sampler = Sampler(alg)
        @test isa(alg, SGLD)
        @test isa(sampler, Sampler{<:Turing.SGLD})
    end
    @numerical_testset "sgld inference" begin
        Random.seed!(125)

        chain = sample(gdemo_default, SGLD(10000, 0.5))

        # Note: samples are weighted by step sizes cf 4.2 in paper
        v = get(chain, [:lf_eps, :s, :m])
        s_res1weightedMean = sum(v.lf_eps .* v.s) / sum(v.lf_eps)
        m_res1weightedMean = sum(v.lf_eps .* v.m) / sum(v.lf_eps)
        @test s_res1weightedMean ≈ 49/24 atol=0.2
        @test m_res1weightedMean ≈ 7/6 atol=0.2
    end
end
