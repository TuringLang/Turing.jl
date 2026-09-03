module MHTests

using AdvancedMH: AdvancedMH
using Distributions: Dirichlet, Exponential, LogNormal, MvNormal, Normal, sample
using DynamicPPL: DynamicPPL, filldist
using LinearAlgebra: I
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @testset, @test_throws
using Turing
using Turing.Inference: Inference

using ..Models: gdemo_default, MoGtest_default
using ..NumericalTests: check_MoGtest_default, check_gdemo

@testset "mh.jl" begin
    @info "Starting MH tests"
    seed = 23

    @testset "mh constructor" begin
        N = 10
        @test_throws ArgumentError MH(:s => Normal())
        @test_throws ArgumentError MH(@varname(s) => filldist(Exponential(1.0), 2))

        sample(gdemo_default, MH(), N)
        sample(gdemo_default, MH([1.0 0.1; 0.1 1.0]), N)
        sample(gdemo_default, Gibbs(:m => MH(), :s => MH()), N)
    end

    @testset "basic accuracy tests" begin
        @testset "linking and Jacobian" begin
            # This model has no likelihood, it's mainly here to test that linking and
            # Jacobians work fine.
            @model function f()
                x ~ Normal()
                return y ~ Beta(2, 2)
            end
            function test_mean_and_std(spl)
                @testset let spl = spl
                    chn = sample(StableRNG(468), f(), spl, 20_000)
                    @test mean(chn[:x]) ≈ mean(Normal()) atol = 0.1
                    @test std(chn[:x]) ≈ std(Normal()) atol = 0.1
                    @test mean(chn[:y]) ≈ mean(Beta(2, 2)) atol = 0.1
                    @test std(chn[:y]) ≈ std(Beta(2, 2)) atol = 0.1
                end
            end
            test_mean_and_std(MH())
            # this uses AdvancedMH
            test_mean_and_std(MH([1.0 0.1; 0.1 1.0]))
        end

        @testset "bad initial parameters" begin
            errmsg = "The initial parameters have zero probability density"

            @model g() = x ~ Beta(2, 2)
            @test_throws errmsg sample(g(), MH(), 2; initial_params=InitFromParams((; x=2)))
        end

        @testset "with dependent priors" begin
            @model function f()
                a ~ Normal()
                x ~ Normal(0.0)
                y ~ Normal(x)
                return 2.0 ~ Normal(y)
            end
            chn = sample(StableRNG(468), f(), MH(), 10000)
            @test mean(chn[:a]) ≈ 0.0 atol = 0.05
            @test mean(chn[:x]) ≈ 2 / 3 atol = 0.05
            @test mean(chn[:y]) ≈ 4 / 3 atol = 0.05
        end
    end

    @testset "chain includes := statements" begin
        @model function f()
            x ~ Normal()
            y := x^2
            return nothing
        end
        for spl in (MH(), MH([1.0;;]))
            chn = sample(f(), spl, 20)
            @test chn[:y] == chn[:x] .^ 2
        end
    end

    @testset "with demo models" begin
        # Set the initial parameters, because if we get unlucky with the initial state,
        # these chains are too short to converge to reasonable numbers.
        discard_initial = 1_000
        initial_params = InitFromParams((s=1.0, m=1.0))

        @testset "gdemo_default" begin
            alg = MH()
            chain = sample(
                StableRNG(seed), gdemo_default, alg, 10_000; discard_initial, initial_params
            )
            check_gdemo(chain; atol=0.1)
        end

        @testset "gdemo_default with MH-within-Gibbs" begin
            alg = Gibbs(:m => MH(), :s => MH())
            chain = sample(
                StableRNG(seed), gdemo_default, alg, 10_000; discard_initial, initial_params
            )
            check_gdemo(chain; atol=0.15)
        end

        @testset "MoGtest_default with Gibbs" begin
            gibbs = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
                @varname(mu1) => MH([1.0;;]),
                @varname(mu2) => MH([1.0;;]),
            )
            initial_params = InitFromParams((mu1=1.0, mu2=1.0, z1=0, z2=0, z3=1, z4=1))
            chain = sample(
                StableRNG(seed),
                MoGtest_default,
                gibbs,
                500;
                discard_initial=100,
                initial_params=initial_params,
            )
            check_MoGtest_default(chain; atol=0.2)
        end
    end

    @testset "with proposal matrix" begin
        mat = [1.0 -0.05; -0.05 1.0]
        spl1 = MH(mat)
        chain1 = sample(StableRNG(seed), gdemo_default, spl1, 2_000)
        check_gdemo(chain1)
    end

    @testset "gibbs MH proposal matrix" begin
        # https://github.com/TuringLang/Turing.jl/issues/1556

        # generate data
        x = rand(Normal(5, 10), 20)
        y = rand(LogNormal(-3, 2), 20)

        # Turing model
        @model function twomeans(x, y)
            # Set Priors
            μ ~ MvNormal(zeros(2), 9 * I)
            σ ~ filldist(Exponential(1), 2)

            # Distributions of supplied data
            x .~ Normal(μ[1], σ[1])
            return y .~ LogNormal(μ[2], σ[2])
        end
        mod = twomeans(x, y)

        # generate covariance matrix for RWMH
        # with small-valued VC matrix to check if we only see very small steps
        vc_μ = convert(Array, 1e-4 * I(2))
        vc_σ = convert(Array, 1e-4 * I(2))
        alg_small = Gibbs(:μ => MH(vc_μ), :σ => MH(vc_σ))
        alg_big = MH()
        chn_small = sample(StableRNG(seed), mod, alg_small, 1_000)
        chn_big = sample(StableRNG(seed), mod, alg_big, 1_000)

        # Test that the small variance version is actually smaller.
        variance_small = var(diff(chn_small[@varname(μ[1])]; dims=1))
        variance_big = var(diff(chn_big[@varname(μ[1])]; dims=1))
        @test variance_small < variance_big / 100.0
    end

    @testset "vector of multivariate distributions" begin
        @model function test(k)
            T = Vector{Vector{Float64}}(undef, k)
            for i in 1:k
                T[i] ~ Dirichlet(5, 1.0)
            end
        end

        chain = sample(StableRNG(seed), test(1), MH(), 5_000)
        for i in 1:5
            @test mean(chain[@varname(T[1][i])]) ≈ 0.2 atol = 0.01
        end

        chain = sample(StableRNG(seed), test(10), MH(), 5_000)
        for j in 1:10, i in 1:5
            @test mean(chain[@varname(T[j][i])]) ≈ 0.2 atol = 0.01
        end
    end

    @testset "LKJCholesky" begin
        for uplo in ['L', 'U']
            @model f() = x ~ LKJCholesky(2, 1, uplo)
            chain = sample(StableRNG(seed), f(), MH(), 5_000)
            indices = [(1, 1), (2, 1), (2, 2)]
            values = [1, 0, 0.785]
            uplo_sym = uplo == 'U' ? :U : :L
            for ((i, j), v) in zip(indices, values)
                if uplo == 'U'  # Transpose
                    @test mean(chain[@varname(x.$uplo_sym[j, i])]) ≈ v atol = 0.01
                else
                    @test mean(chain[@varname(x.$uplo_sym[i, j])]) ≈ v atol = 0.01
                end
            end
        end
    end
end

end
