using Turing
using Turing: WelfordCovar, WelfordVar, NaiveCovar, add_sample!,
    get_covar, get_var, reset!
using Test, LinearAlgebra, Random

include("../../test_utils/AllUtils.jl")

@testset "adapt.jl" begin
    @turing_testset "covariance estimator" begin
        let
            D = 1000
            wc = WelfordCovar(0, zeros(D), zeros(D,D))
            add_sample!(wc, randn(D))
            reset!(wc)

            # Check that reseting zeros everything.
            @test wc.n === 0
            @test wc.μ == zeros(D)
            @test wc.M == zeros(D,D)

            # Ensure that asking for the variance doesn't mutate the WelfordVar.
            add_sample!(wc, randn(D))
            add_sample!(wc, randn(D))
            μ, M = deepcopy(wc.μ), deepcopy(wc.M)

            @test wc.μ == μ
            @test wc.M == M
        end

        # Check that the estimated variance is approximately correct.
        let
            D = 10
            wc = WelfordCovar(0, zeros(D), zeros(D,D))

            for _ = 1:10000
                s = randn(D)
                add_sample!(wc, s)
            end

            covar = get_covar(wc)

            @test covar ≈ LinearAlgebra.diagm(0 => ones(D)) atol=0.2
        end
    end
    @turing_testset "variance estimator" begin
        let
            D = 1000
            wv = WelfordVar(0, zeros(D), zeros(D))
            add_sample!(wv, randn(D))
            reset!(wv)

            # Check that reseting zeros everything.
            @test wv.n === 0
            @test wv.μ == zeros(D)
            @test wv.M == zeros(D)

            # Ensure that asking for the variance doesn't mutate the WelfordVar.
            add_sample!(wv, randn(D))
            add_sample!(wv, randn(D))
            μ, M = deepcopy(wv.μ), deepcopy(wv.M)

            @test wv.μ == μ
            @test wv.M == M
        end

        # Check that the estimated variance is approximately correct.
        let
            D = 10
            wv = WelfordVar(0, zeros(D), zeros(D))

            for _ = 1:10000
                s = randn(D)
                add_sample!(wv, s)
            end

            var = get_var(wv)

            @test var ≈ ones(D) atol=0.1
        end
    end
end
