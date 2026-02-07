module MHTests

using AdvancedMH: AdvancedMH
using Distributions:
    Bernoulli, Dirichlet, Exponential, InverseGamma, LogNormal, MvNormal, Normal, sample
using DynamicPPL: DynamicPPL
using LinearAlgebra: I
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @testset, @test_throws
using Turing
using Turing.Inference: Inference

using ..Models: gdemo_default, MoGtest_default
using ..NumericalTests: check_MoGtest_default, check_gdemo, check_numerical

GKernel(variance, vn) = (vnt -> Normal(vnt[vn], sqrt(variance)))

@testset "mh.jl" begin
    @info "Starting MH tests"
    seed = 23

    @testset "mh constructor" begin
        N = 10
        s1 = MH(:s => InverseGamma(2, 3), :m => GKernel(3.0, @varname(m)))
        s2 = MH()
        s3 = MH([1.0 0.1; 0.1 1.0])

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)

        s4 = Gibbs(:m => MH(), :s => MH())
        c4 = sample(gdemo_default, s4, N)
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
            test_mean_and_std(MH(@varname(x) => Normal(1.0)))
            test_mean_and_std(MH(@varname(y) => Uniform(0, 1)))
            test_mean_and_std(MH(@varname(x) => Normal(1.0), @varname(y) => Uniform(0, 1)))
            test_mean_and_std(MH(@varname(x) => LinkedRW(0.5)))
            test_mean_and_std(MH(@varname(y) => LinkedRW(0.5)))
            # this is a random walk in unlinked space
            test_mean_and_std(MH(@varname(y) => vnt -> Normal(vnt[@varname(y)], 0.5)))
            test_mean_and_std(MH(@varname(x) => Normal(), @varname(y) => LinkedRW(0.5)))
            test_mean_and_std(MH(@varname(x) => LinkedRW(0.5), @varname(y) => Normal()))
            # this uses AdvancedMH
            test_mean_and_std(MH([1.0 0.1; 0.1 1.0]))
        end

        @testset "bad proposals" begin
            errmsg = "The initial parameters have zero probability density"

            @model f() = x ~ Normal()
            # Here we give `x` a constrained proposal. Any samples of `x` that fall outside
            # of it will get a proposal density of -Inf, so should be rejected
            fspl = MH(@varname(x) => Uniform(-1, 1))
            # We now start the chain outside the proposal region. The point of this test is 
            # to make sure that we throw a sensible error.
            @test_throws errmsg sample(f(), fspl, 2; initial_params=InitFromParams((; x=2)))

            # Same here, except that now it's the proposal that's bad, not the initial
            # parameters.
            @model g() = x ~ Beta(2, 2)
            gspl = MH(@varname(x) => Uniform(-2, -1))
            @test_throws errmsg sample(g(), gspl, 2; initial_params=InitFromPrior())
        end

        @testset "with unspecified priors that depend on other variables" begin
            @model function f()
                a ~ Normal()
                x ~ Normal(0.0)
                y ~ Normal(x)
                return 2.0 ~ Normal(y)
            end
            # If we don't specify a proposal for `y`, it will be sampled from `Normal(x)`.
            # However, we need to be careful here since the value of `x` varies! This testset is
            # essentially a test to check that `MHUnspecifiedPriorAccumulator` is doing
            # the right thing, i.e., it correctly accumulates the prior for the evaluation that
            # we're interested in.
            chn = sample(StableRNG(468), f(), MH(@varname(a) => Normal()), 10000)
            @test mean(chn[:a]) ≈ 0.0 atol = 0.05
            @test mean(chn[:x]) ≈ 2 / 3 atol = 0.05
            @test mean(chn[:y]) ≈ 4 / 3 atol = 0.05

            # Note that if we additionally don't specify a proposal for `a`, then it will defer
            # to InitFromPrior for all variables, and its prior *is* Normal(). So we can do
            # this:
            chn2 = sample(StableRNG(468), f(), MH(), 10000)
            # and it should actually give us exactly the same results as above.
            #
            # This test *does* depend on internal implementation details of MH and is a bit
            # brittle. If it fails, you should check that the values are still approximately
            # correct (i.e. 0, 2/3, and 4/3). If they are still correct, then it probably just
            # means that the internal implementation details have changed, and this test can
            # just be updated (or removed).
            @test mean(chn2[:a]) ≈ mean(chn[:a])
            @test mean(chn2[:x]) ≈ mean(chn[:x])
            @test mean(chn2[:y]) ≈ mean(chn[:y])
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

        @testset "gdemo_default with custom proposals" begin
            alg = MH(:s => InverseGamma(2, 3), :m => GKernel(1.0, @varname(m)))
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
                @varname(mu1) => MH(:mu1 => GKernel(1, @varname(mu1))),
                @varname(mu2) => MH(:mu2 => GKernel(1, @varname(mu2))),
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
        variance_small = var(diff(Array(chn_small["μ[1]"]); dims=1))
        variance_big = var(diff(Array(chn_big["μ[1]"]); dims=1))
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
            @test mean(chain, "T[1][$i]") ≈ 0.2 atol = 0.01
        end

        chain = sample(StableRNG(seed), test(10), MH(), 5_000)
        for j in 1:10, i in 1:5
            @test mean(chain, "T[$j][$i]") ≈ 0.2 atol = 0.01
        end
    end

    @testset "LKJCholesky" begin
        for uplo in ['L', 'U']
            @model f() = x ~ LKJCholesky(2, 1, uplo)
            chain = sample(StableRNG(seed), f(), MH(), 5_000)
            indices = [(1, 1), (2, 1), (2, 2)]
            values = [1, 0, 0.785]
            for ((i, j), v) in zip(indices, values)
                if uplo == 'U'  # Transpose
                    @test mean(chain, "x.$uplo[$j, $i]") ≈ v atol = 0.01
                else
                    @test mean(chain, "x.$uplo[$i, $j]") ≈ v atol = 0.01
                end
            end
        end
    end
end

end
