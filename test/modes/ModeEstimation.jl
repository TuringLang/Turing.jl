using Turing
using Optim
using Test
using StatsBase
using NamedArrays
using ReverseDiff
using Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "ModeEstimation.jl" begin
    @testset "MLE" begin
        Random.seed!(222)
        
        m1 = optimize(gdemo_default, MLE())
        m2 = optimize(gdemo_default, MLE(), NelderMead())
        m3 = optimize(gdemo_default, MLE(), Newton())

        @test all(isapprox.(m1.values.array - [0.0625031, 1.75], 0.0, atol=0.01))
        @test all(isapprox.(m2.values.array - [0.0625031, 1.75], 0.0, atol=0.01))
        @test all(isapprox.(m3.values.array - [0.0625031, 1.75], 0.0, atol=0.01))
    end
    @testset "MAP" begin
        Random.seed!(222)
        
        m1 = optimize(gdemo_default, MAP())
        m2 = optimize(gdemo_default, MAP(), NelderMead())
        m3 = optimize(gdemo_default, MAP(), Newton())
        
        @test all(isapprox.(m1.values.array - [49/24, 7/6], 0.0, atol=0.01))
        @test all(isapprox.(m2.values.array - [49/24, 7/6], 0.0, atol=0.01))
        @test all(isapprox.(m3.values.array - [49/24, 7/6], 0.0, atol=0.01))
    end

    @testset "StatsBase integration" begin
        Random.seed!(54321)
        mle_est = optimize(gdemo_default, MLE())

        @test coefnames(mle_est) == ["s", "m"]

        diffs = coef(mle_est).array - [0.0625031; 1.75001]
        @test all(isapprox.(diffs, 0.0, atol=0.1))

        infomat = [0.003907027690416608 4.157954948417027e-7; 4.157954948417027e-7 0.03125155528962335]
        @test all(isapprox.(infomat - informationmatrix(mle_est), 0.0, atol=0.01))

        ctable = coeftable(mle_est)
        @test ctable isa StatsBase.CoefTable

        s = stderror(mle_est).array
        @test all(isapprox.(s - [0.06250415643292194, 0.17677963626053916], 0.0, atol=0.01))

        @test coefnames(mle_est) == params(mle_est)
        @test vcov(mle_est) == informationmatrix(mle_est)

        @test isapprox(loglikelihood(mle_est), -0.0652883561466624, atol=0.01)
    end
end