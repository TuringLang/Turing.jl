using Turing, Random, Test
using Turing: Sampler

include("../test_utils/AllUtils.jl")

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
