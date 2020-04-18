using Turing, Random, Test
import Turing.Inference
import AdvancedMH

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "mh.jl" begin
    @turing_testset "mh constructor" begin
        Random.seed!(0)
        N = 500
        s1 = MH(
            (:s, InverseGamma(2,3)),
            (:m, GKernel(3.0)))
        s2 = MH(:s, :m)
        s3 = MH()
        s4 = Gibbs(MH(:m), MH(:s))

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
    end
    @numerical_testset "mh inference" begin
        Random.seed!(125)
        alg = MH()
        chain = sample(gdemo_default, alg, 2000)
        check_gdemo(chain, atol = 0.1)

        # MH with Gaussian proposal
        alg = MH(
            (:s, InverseGamma(2,3)),
            (:m, GKernel(1.0)))
        chain = sample(gdemo_default, alg, 7000)
        check_gdemo(chain, atol = 0.1)

        # MH within Gibbs
        alg = Gibbs(MH(:m), MH(:s))
        chain = sample(gdemo_default, alg, 2000)
        check_gdemo(chain, atol = 0.1)

        # MoGtest
        gibbs = Gibbs(
            CSMC(15, :z1, :z2, :z3, :z4),
            MH((:mu1,GKernel(1)), (:mu2,GKernel(1)))
        )
        chain = sample(MoGtest_default, gibbs, 5000)
        check_MoGtest_default(chain, atol = 0.15)
    end

    # Test MH shape passing.
    @turing_testset "shape" begin
        @model M(mu, sigma, observable) = begin
            z ~ MvNormal(mu, sigma)

            m = Array{Float64}(undef, 1, 2)
            m[1] ~ Normal(0, 1)
            m[2] ~ InverseGamma(2, 1)
            s ~ InverseGamma(2, 1)

            observable ~ Bernoulli(cdf(Normal(), z' * z))

            1.5 ~ Normal(m[1], m[2])
            -1.5 ~ Normal(m[1], m[2])

            1.5 ~ Normal(m[1], s)
            2.0 ~ Normal(m[1], s)
        end

        model = M(zeros(2), ones(2), 1)
        sampler = Inference.Sampler(MH(), model)

        dt, vt = Inference.dist_val_tuple(sampler)

        @test dt[:z] isa AdvancedMH.StaticProposal{<:MvNormal}
        @test dt[:m] isa AdvancedMH.StaticProposal{Vector{ContinuousUnivariateDistribution}}
        @test dt[:m].proposal[1] isa Normal && dt[:m].proposal[2] isa InverseGamma
        @test dt[:s] isa AdvancedMH.StaticProposal{<:InverseGamma}

        @test vt[:z] isa Vector{Float64} && length(vt[:z]) == 2
        @test vt[:m] isa Vector{Float64} && length(vt[:m]) == 2
        @test vt[:s] isa Float64

        chain = sample(model, MH(), 100)

        @test chain isa MCMCChains.Chains
    end
end
