using Turing, Random, Test
using Turing: Sampler

include("../test_utils/AllUtils.jl")

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
