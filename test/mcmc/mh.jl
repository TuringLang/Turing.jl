module MHTests

using AdvancedMH: AdvancedMH
using Distributions:
    Bernoulli, Dirichlet, Exponential, InverseGamma, LogNormal, MvNormal, Normal, sample
using DynamicPPL: DynamicPPL
using LinearAlgebra: I
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @testset
using Turing
using Turing.Inference: Inference

using ..Models: gdemo_default, MoGtest_default
using ..NumericalTests: check_MoGtest_default, check_gdemo, check_numerical

GKernel(var) = (x) -> Normal(x, sqrt.(var))

@testset "mh.jl" begin
    @info "Starting MH tests"
    seed = 23

    @testset "mh constructor" begin
        N = 10
        s1 = MH((:s, InverseGamma(2, 3)), (:m, GKernel(3.0)))
        s2 = MH(:s => InverseGamma(2, 3), :m => GKernel(3.0))
        s3 = MH()
        s4 = MH([1.0 0.1; 0.1 1.0])

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)

        s5 = Gibbs(:m => MH(), :s => MH())
        c5 = sample(gdemo_default, s5, N)

        # s6 = externalsampler(MH(gdemo_default, proposal_type=AdvancedMH.RandomWalkProposal))
        # c6 = sample(gdemo_default, s6, N)

        # NOTE: Broken because MH doesn't really follow the `logdensity` interface, but calls
        # it with `NamedTuple` instead of `AbstractVector`.
        # s7 = externalsampler(MH(gdemo_default, proposal_type=AdvancedMH.StaticProposal))
        # c7 = sample(gdemo_default, s7, N)
    end

    @testset "mh inference" begin
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
            alg = MH((:s, InverseGamma(2, 3)), (:m, GKernel(1.0)))
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
                @varname(mu1) => MH((:mu1, GKernel(1))),
                @varname(mu2) => MH((:mu2, GKernel(1))),
            )
            initial_params = InitFromParams((
                mu1=1.0, mu2=1.0, z1=0.0, z2=0.0, z3=1.0, z4=1.0
            ))
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

    # Test MH shape passing.
    @testset "shape" begin
        @model function M(mu, sigma, observable)
            z ~ MvNormal(mu, sigma)

            m = Array{Float64}(undef, 1, 2)
            m[1] ~ Normal(0, 1)
            m[2] ~ InverseGamma(2, 1)
            s ~ InverseGamma(2, 1)

            observable ~ Bernoulli(cdf(Normal(), z' * z))

            1.5 ~ Normal(m[1], m[2])
            -1.5 ~ Normal(m[1], m[2])

            1.5 ~ Normal(m[1], s)
            return 2.0 ~ Normal(m[1], s)
        end

        model = M(zeros(2), I, 1)
        sampler = MH()

        dt, vt = Inference.dist_val_tuple(sampler, DynamicPPL.VarInfo(model))

        @test dt[:z] isa AdvancedMH.StaticProposal{false,<:MvNormal}
        @test dt[:m] isa
            AdvancedMH.StaticProposal{false,Vector{ContinuousUnivariateDistribution}}
        @test dt[:m].proposal[1] isa Normal && dt[:m].proposal[2] isa InverseGamma
        @test dt[:s] isa AdvancedMH.StaticProposal{false,<:InverseGamma}

        @test vt[:z] isa Vector{Float64} && length(vt[:z]) == 2
        @test vt[:m] isa Vector{Float64} && length(vt[:m]) == 2
        @test vt[:s] isa Float64

        chain = sample(model, MH(), 10)

        @test chain isa MCMCChains.Chains
    end

    @testset "proposal matrix" begin
        mat = [1.0 -0.05; -0.05 1.0]

        prop1 = mat # Matrix only constructor
        prop2 = AdvancedMH.RandomWalkProposal(MvNormal(mat)) # Explicit proposal constructor

        spl1 = MH(prop1)
        spl2 = MH(prop2)

        # Test that the two constructors are equivalent.
        @test spl1.proposals.proposal.μ == spl2.proposals.proposal.μ
        @test spl1.proposals.proposal.Σ.mat == spl2.proposals.proposal.Σ.mat

        # Test inference.
        chain1 = sample(StableRNG(seed), gdemo_default, spl1, 2_000)
        chain2 = sample(StableRNG(seed), gdemo_default, spl2, 2_000)

        check_gdemo(chain1)
        check_gdemo(chain2)
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
        alg_small = Gibbs(:μ => MH((:μ, vc_μ)), :σ => MH((:σ, vc_σ)))
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

    @testset "MH link/invlink" begin
        vi_base = DynamicPPL.VarInfo(gdemo_default)

        # Don't link when no proposals are given since we're using priors
        # as proposals.
        vi = deepcopy(vi_base)
        spl = MH()
        vi = Turing.Inference.maybe_link!!(vi, spl, spl.proposals, gdemo_default)
        @test !DynamicPPL.is_transformed(vi)

        # Link if proposal is `AdvancedHM.RandomWalkProposal`
        vi = deepcopy(vi_base)
        d = length(vi_base[:])
        spl = MH(AdvancedMH.RandomWalkProposal(MvNormal(zeros(d), I)))
        vi = Turing.Inference.maybe_link!!(vi, spl, spl.proposals, gdemo_default)
        @test DynamicPPL.is_transformed(vi)

        # Link if ALL proposals are `AdvancedHM.RandomWalkProposal`.
        vi = deepcopy(vi_base)
        spl = MH(:s => AdvancedMH.RandomWalkProposal(Normal()))
        vi = Turing.Inference.maybe_link!!(vi, spl, spl.proposals, gdemo_default)
        @test DynamicPPL.is_transformed(vi)

        # Don't link if at least one proposal is NOT `RandomWalkProposal`.
        # TODO: make it so that only those that are using `RandomWalkProposal`
        # are linked! I.e. resolve https://github.com/TuringLang/Turing.jl/issues/1583.
        # https://github.com/TuringLang/Turing.jl/pull/1582#issuecomment-817148192
        vi = deepcopy(vi_base)
        spl = MH(
            :m => AdvancedMH.StaticProposal(Normal()),
            :s => AdvancedMH.RandomWalkProposal(Normal()),
        )
        vi = Turing.Inference.maybe_link!!(vi, spl, spl.proposals, gdemo_default)
        @test !DynamicPPL.is_transformed(vi)
    end

    @testset "`filldist` proposal (issue #2180)" begin
        @model demo_filldist_issue2180() = x ~ MvNormal(zeros(3), I)
        chain = sample(
            StableRNG(seed),
            demo_filldist_issue2180(),
            MH(AdvancedMH.RandomWalkProposal(filldist(Normal(), 3))),
            10_000,
        )
        check_numerical(
            chain, [Symbol("x[1]"), Symbol("x[2]"), Symbol("x[3]")], [0, 0, 0]; atol=0.2
        )
    end
end

end
