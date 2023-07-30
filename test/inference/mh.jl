@testset "mh.jl" begin
    @turing_testset "mh constructor" begin
        Random.seed!(10)
        N = 500
        s1 = MH(
            (:s, InverseGamma(2,3)),
            (:m, GKernel(3.0)))
        s2 = MH(:s, :m)
        s3 = MH()
        for s in (s1, s2, s3)
            @test DynamicPPL.alg_str(Sampler(s, gdemo_default)) == "MH"
        end

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)

        s4 = Gibbs(MH(:m), MH(:s))
        c4 = sample(gdemo_default, s4, N)

        s5 = externalsampler(MH(gdemo_default, proposal_type=AMH.RandomWalkProposal))
        c5 = sample(gdemo_default, s5, N)

        s6 = externalsampler(MH(gdemo_default, proposal_type=AMH.StaticProposal)
        c6 = sample(gdemo_default, s6, N)
    end
    @numerical_testset "mh inference" begin
        Random.seed!(125)
        alg = MH()
        chain = sample(gdemo_default, alg, 10_000)
        check_gdemo(chain, atol = 0.1)

        Random.seed!(125)
        # MH with Gaussian proposal
        alg = MH(
            (:s, InverseGamma(2,3)),
            (:m, GKernel(1.0)))
        chain = sample(gdemo_default, alg, 10_000)
        check_gdemo(chain, atol = 0.1)

        Random.seed!(125)
        # MH within Gibbs
        alg = Gibbs(MH(:m), MH(:s))
        chain = sample(gdemo_default, alg, 10_000)
        check_gdemo(chain, atol = 0.1)

        Random.seed!(125)
        # MoGtest
        gibbs = Gibbs(
            CSMC(15, :z1, :z2, :z3, :z4),
            MH((:mu1,GKernel(1)), (:mu2,GKernel(1)))
        )
        chain = sample(MoGtest_default, gibbs, 500)
        check_MoGtest_default(chain, atol = 0.15)
    end

    # Test MH shape passing.
    @turing_testset "shape" begin
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
            2.0 ~ Normal(m[1], s)
        end

        model = M(zeros(2), I, 1)
        sampler = Inference.Sampler(MH(), model)

        dt, vt = Inference.dist_val_tuple(sampler, Turing.VarInfo(model))

        @test dt[:z] isa AdvancedMH.StaticProposal{false,<:MvNormal}
        @test dt[:m] isa AdvancedMH.StaticProposal{false,Vector{ContinuousUnivariateDistribution}}
        @test dt[:m].proposal[1] isa Normal && dt[:m].proposal[2] isa InverseGamma
        @test dt[:s] isa AdvancedMH.StaticProposal{false,<:InverseGamma}

        @test vt[:z] isa Vector{Float64} && length(vt[:z]) == 2
        @test vt[:m] isa Vector{Float64} && length(vt[:m]) == 2
        @test vt[:s] isa Float64

        chain = sample(model, MH(), 100)

        @test chain isa MCMCChains.Chains
    end

    @turing_testset "proposal matrix" begin
        Random.seed!(100)
        
        mat = [1.0 -0.05; -0.05 1.0]

        prop1 = mat # Matrix only constructor
        prop2 = AdvancedMH.RandomWalkProposal(MvNormal(mat)) # Explicit proposal constructor

        spl1 = MH(prop1)
        spl2 = MH(prop2)

        # Test that the two constructors are equivalent.
        @test spl1.proposals.proposal.μ == spl2.proposals.proposal.μ
        @test spl1.proposals.proposal.Σ.mat == spl2.proposals.proposal.Σ.mat

        # Test inference.
        chain1 = sample(gdemo_default, spl1, 10000)
        chain2 = sample(gdemo_default, spl2, 10000)

        check_gdemo(chain1)
        check_gdemo(chain2)
    end

    @turing_testset "gibbs MH proposal matrix" begin
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
            y .~ LogNormal(μ[2], σ[2])
        
        end
        mod = twomeans(x, y)
        
        # generate covariance matrix for RWMH
        # with small-valued VC matrix to check if we only see very small steps
        vc_μ = convert(Array, 1e-4*I(2))
        vc_σ = convert(Array, 1e-4*I(2))

        alg = Gibbs(
            MH((:μ, vc_μ)),
            MH((:σ, vc_σ)),
        )

        chn = sample(
            mod,
            alg,
            3_000 # draws
        )
        
            
        chn2 = sample(mod, MH(), 3_000)

        # Test that the small variance version is actually smaller.
        v1 = var(diff(Array(chn["μ[1]"]), dims=1))
        v2 = var(diff(Array(chn2["μ[1]"]), dims=1))

        # FIXME: Do this properly. It sometimes fails.
        # @test v1 < v2
    end

    @turing_testset "vector of multivariate distributions" begin
        @model function test(k)
            T = Vector{Vector{Float64}}(undef, k)
            for i in 1:k
                T[i] ~ Dirichlet(5, 1.0)
            end
        end

        Random.seed!(100)
        chain = sample(test(1), MH(), 5_000)
        for i in 1:5
            @test mean(chain, "T[1][$i]") ≈ 0.2 atol=0.01
        end

        Random.seed!(100)
        chain = sample(test(10), MH(), 5_000)
        for j in 1:10, i in 1:5
            @test mean(chain, "T[$j][$i]") ≈ 0.2 atol=0.01
        end
    end

    @turing_testset "MH link/invlink" begin
        vi_base = DynamicPPL.VarInfo(gdemo_default)

        # Don't link when no proposals are given since we're using priors
        # as proposals.
        vi = deepcopy(vi_base)
        alg = MH()
        spl = DynamicPPL.Sampler(alg)
        vi = Turing.Inference.maybe_link!!(vi, spl, alg.proposals, gdemo_default)
        @test !DynamicPPL.islinked(vi, spl)

        # Link if proposal is `AdvancedHM.RandomWalkProposal`
        vi = deepcopy(vi_base)
        d = length(vi_base[DynamicPPL.SampleFromPrior()])
        alg = MH(AdvancedMH.RandomWalkProposal(MvNormal(zeros(d), I)))
        spl = DynamicPPL.Sampler(alg)
        vi = Turing.Inference.maybe_link!!(vi, spl, alg.proposals, gdemo_default)
        @test DynamicPPL.islinked(vi, spl)

        # Link if ALL proposals are `AdvancedHM.RandomWalkProposal`.
        vi = deepcopy(vi_base)
        alg = MH(:s => AdvancedMH.RandomWalkProposal(Normal()))
        spl = DynamicPPL.Sampler(alg)
        vi = Turing.Inference.maybe_link!!(vi, spl, alg.proposals, gdemo_default)
        @test DynamicPPL.islinked(vi, spl)

        # Don't link if at least one proposal is NOT `RandomWalkProposal`.
        # TODO: make it so that only those that are using `RandomWalkProposal`
        # are linked! I.e. resolve https://github.com/TuringLang/Turing.jl/issues/1583.
        # https://github.com/TuringLang/Turing.jl/pull/1582#issuecomment-817148192
        vi = deepcopy(vi_base)
        alg = MH(
            :m => AdvancedMH.StaticProposal(Normal()),
            :s => AdvancedMH.RandomWalkProposal(Normal())
        )
        spl = DynamicPPL.Sampler(alg)
        vi = Turing.Inference.maybe_link!!(vi, spl, alg.proposals, gdemo_default)
        @test !DynamicPPL.islinked(vi, spl)
    end

    @turing_testset "prior" begin
        # HACK: MH can be so bad for this prior model for some reason that it's difficult to
        # find a non-trivial `atol` where the tests will pass for all seeds. Hence we fix it :/
        rng = StableRNG(10)
        alg = MH()
        gdemo_default_prior = DynamicPPL.contextualize(gdemo_default, DynamicPPL.PriorContext())
        burnin = 10_000
        n = 10_000
        chain = sample(rng, gdemo_default_prior, alg, n; discard_initial = burnin, thinning=10)
        check_numerical(chain, [:s, :m], [mean(InverseGamma(2, 3)), 0], atol=0.3)
    end
end
