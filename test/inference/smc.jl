using Turing, Random, Test
using AdvancedSMC
using Turing.Inference: getspace

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "SMC" begin
    @turing_testset "constructor" begin
        s = SMC()
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === ()

        s = SMC(:x)
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = SMC((:x,))
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = SMC(:x, :y)
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = SMC((:x, :y))
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = SMC(0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === ()

        s = SMC(0.6, (:x,))
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === (:x,)

        s = SMC(resample_multinomial, 0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === ()

        s = SMC(resample_multinomial, 0.6, (:x,))
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === (:x,)

        s = SMC(resample_systematic)
        @test s.resampler === resample_systematic
        @test getspace(s) === ()

        s = SMC(resample_systematic, (:x,))
        @test s.resampler === resample_systematic
        @test getspace(s) === (:x,)
    end

    @turing_testset "models" begin
        @model function normal()
            a ~ Normal(4,5)
            3 ~ Normal(a,2)
            b ~ Normal(a,1)
            1.5 ~ Normal(b,2)
            a, b
        end

        tested = sample(normal(), SMC(), 100);

        # failing test
        @model function fail_smc()
            a ~ Normal(4,5)
            3 ~ Normal(a,2)
            b ~ Normal(a,1)
            if a >= 4.0
                1.5 ~ Normal(b,2)
            end
            a, b
        end

        @test_throws ErrorException sample(fail_smc(), SMC(), 100)
    end

    @turing_testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            x
        end

        chains_smc = sample(test(), SMC(), 100)

        @test all(isone, chains_smc[:x].value)
        @test chains_smc.logevidence ≈ -2 * log(2)
    end
end

@testset "PG" begin
    @turing_testset "constructor" begin
        s = PG(10)
        @test s.nparticles == 10
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === ()

        s = PG(20, :x)
        @test s.nparticles == 20
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(30, (:x,))
        @test s.nparticles == 30
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(40, :x, :y)
        @test s.nparticles == 40
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(50, (:x, :y))
        @test s.nparticles == 50
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(60, 0.6)
        @test s.nparticles == 60
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === ()

        s = PG(70, 0.6, (:x,))
        @test s.nparticles == 70
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === (:x,)

        s = PG(80, resample_multinomial, 0.6)
        @test s.nparticles == 80
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === ()

        s = PG(90, resample_multinomial, 0.6, (:x,))
        @test s.nparticles == 90
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === (:x,)

        s = PG(100, resample_systematic)
        @test s.nparticles == 100
        @test s.resampler === resample_systematic
        @test getspace(s) === ()

        s = PG(110, resample_systematic, (:x,))
        @test s.nparticles == 110
        @test s.resampler === resample_systematic
        @test getspace(s) === (:x,)
    end

    @turing_testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            x
        end

        chains_pg = sample(test(), PG(10), 100)

        @test all(isone, chains_pg[:x].value)
        @test chains_pg.logevidence ≈ -2 * log(2) atol = 0.01
    end
end
