using Turing, Random, Test
using Turing.Inference: getspace
using AdvancedPS # TODO: maybe use import instead?
# import AdvancedPS

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")


@testset "PG" begin
    @turing_testset "constructor" begin
        s = PG(10)
        @test s.nparticles == 10
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === ()

        s = PG(20, :x)
        @test s.nparticles == 20
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(30, (:x,))
        @test s.nparticles == 30
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(40, :x, :y)
        @test s.nparticles == 40
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(50, (:x, :y))
        @test s.nparticles == 50
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(60, 0.6)
        @test s.nparticles == 60
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)
        @test getspace(s) === ()

        s = PG(70, 0.6, (:x,))
        @test s.nparticles == 70
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)
        @test getspace(s) === (:x,)

        s = PG(80, AdvancedPS.resample_multinomial, 0.6)
        @test s.nparticles == 80
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)
        @test getspace(s) === ()

        s = PG(90, AdvancedPS.resample_multinomial, 0.6, (:x,))
        @test s.nparticles == 90
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)
        @test getspace(s) === (:x,)

        s = PG(100, AdvancedPS.resample_systematic)
        @test s.nparticles == 100
        @test s.resampler === AdvancedPS.resample_systematic
        @test getspace(s) === ()

        s = PG(110, AdvancedPS.resample_systematic, (:x,))
        @test s.nparticles == 110
        @test s.resampler === AdvancedPS.resample_systematic
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