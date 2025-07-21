module ParticleMCMCTests

using ..Models: gdemo_default
#using ..Models: MoGtest, MoGtest_default
using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using Distributions: Bernoulli, Beta, Gamma, Normal, sample
using Random: Random
using Test: @test, @test_throws, @testset
using Turing

@testset "SMC" begin
    @testset "constructor" begin
        s = SMC()
        @test s.resampler == ResampleWithESSThreshold()

        s = SMC(0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)

        s = SMC(resample_multinomial, 0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)

        s = SMC(resample_systematic)
        @test s.resampler === resample_systematic
    end

    @testset "models" begin
        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        tested = sample(normal(), SMC(), 100)

        # TODO(mhauru) This needs an explanation for why it fails.
        # failing test
        @model function fail_smc()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            if a >= 4.0
                1.5 ~ Normal(b, 2)
            end
            return a, b
        end

        @test_throws ErrorException sample(fail_smc(), SMC(), 100)
    end

    @testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains_smc = sample(test(), SMC(), 100)

        @test all(isone, chains_smc[:x])
        @test chains_smc.logevidence ≈ -2 * log(2)
    end
end

@testset "PG" begin
    @testset "constructor" begin
        s = PG(10)
        @test s.nparticles == 10
        @test s.resampler == ResampleWithESSThreshold()

        s = PG(60, 0.6)
        @test s.nparticles == 60
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)

        s = PG(80, resample_multinomial, 0.6)
        @test s.nparticles == 80
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)

        s = PG(100, resample_systematic)
        @test s.nparticles == 100
        @test s.resampler === resample_systematic
    end

    @testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains_pg = sample(test(), PG(10), 100)

        @test all(isone, chains_pg[:x])
        @test chains_pg.logevidence ≈ -2 * log(2) atol = 0.01
    end

    # https://github.com/TuringLang/Turing.jl/issues/1598
    @testset "reference particle" begin
        c = sample(gdemo_default, PG(1), 1_000)
        @test length(unique(c[:m])) == 1
        @test length(unique(c[:s])) == 1
    end

    # https://github.com/TuringLang/Turing.jl/issues/2007
    @testset "keyword arguments not supported" begin
        @model kwarg_demo(; x=2) = return x
        @test_throws ErrorException sample(kwarg_demo(), PG(1), 10)
    end
end

end
