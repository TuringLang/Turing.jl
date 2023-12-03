@testset "Testing gibbs.jl with $adbackend" for adbackend in (AutoForwardDiff(; chunksize=0), AutoReverseDiff(false))
    @turing_testset "gibbs constructor" begin
        N = 500
        s1 = Gibbs(HMC(0.1, 5, :s, :m; adtype=adbackend))
        s2 = Gibbs(PG(10, :s, :m))
        s3 = Gibbs(PG(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        s4 = Gibbs(PG(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        s5 = Gibbs(CSMC(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        s6 = Gibbs(HMC(0.1, 5, :s; adtype=adbackend), ESS(:m))
        for s in (s1, s2, s3, s4, s5, s6)
            @test DynamicPPL.alg_str(Turing.Sampler(s, gdemo_default)) == "Gibbs"
        end

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
        c5 = sample(gdemo_default, s5, N)
        c6 = sample(gdemo_default, s6, N)

        # Test gid of each samplers
        g = Turing.Sampler(s3, gdemo_default)

        _, state = AbstractMCMC.step(Random.default_rng(), gdemo_default, g)
        @test state.samplers[1].selector != g.selector
        @test state.samplers[2].selector != g.selector
        @test state.samplers[1].selector != state.samplers[2].selector

        # run sampler: progress logging should be disabled and
        # it should return a Chains object
        @test sample(gdemo_default, g, N) isa MCMCChains.Chains
    end
    @numerical_testset "gibbs inference" begin
        Random.seed!(100)
        alg = Gibbs(CSMC(15, :s), HMC(0.2, 4, :m; adtype=adbackend))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol=0.15)

        Random.seed!(100)

        alg = Gibbs(MH(:s), HMC(0.2, 4, :m; adtype=adbackend))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol=0.1)

        alg = Gibbs(CSMC(15, :s), ESS(:m))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol=0.1)

        alg = CSMC(15)
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol=0.1)

        Random.seed!(200)
        gibbs = Gibbs(PG(15, :z1, :z2, :z3, :z4), HMC(0.15, 3, :mu1, :mu2; adtype=adbackend))
        chain = sample(MoGtest_default, gibbs, 10_000)
        check_MoGtest_default(chain, atol=0.15)

        Random.seed!(200)
        for alg in [
            Gibbs((MH(:s), 2), (HMC(0.2, 4, :m; adtype=adbackend), 1)),
            Gibbs((MH(:s), 1), (HMC(0.2, 4, :m; adtype=adbackend), 2)),
        ]
            chain = sample(gdemo(1.5, 2.0), alg, 10_000)
            check_gdemo(chain; atol=0.15)
        end
    end

    @turing_testset "transitions" begin
        @model function gdemo_copy()
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            1.5 ~ Normal(m, sqrt(s))
            2.0 ~ Normal(m, sqrt(s))
            return s, m
        end
        model = gdemo_copy()

        function AbstractMCMC.bundle_samples(
            samples::Vector,
            ::DynamicPPL.Model,
            ::Turing.Sampler{<:Gibbs},
            state,
            ::Type{MCMCChains.Chains};
            kwargs...
        )
            samples isa Vector{<:Inference.Transition} ||
                error("incorrect transitions")
            return
        end

        function callback(rng, model, sampler, sample, state, i; kwargs...)
            sample isa Inference.Transition || error("incorrect sample")
            return
        end

        alg = Gibbs(MH(:s), HMC(0.2, 4, :m; adtype=adbackend))
        sample(model, alg, 100; callback=callback)
    end
    @turing_testset "dynamic model" begin
        @model function imm(y, alpha, ::Type{M}=Vector{Float64}) where {M}
            N = length(y)
            rpm = DirichletProcess(alpha)

            z = zeros(Int, N)
            cluster_counts = zeros(Int, N)
            fill!(cluster_counts, 0)

            for i in 1:N
                z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
                cluster_counts[z[i]] += 1
            end

            Kmax = findlast(!iszero, cluster_counts)
            m = M(undef, Kmax)
            for k = 1:Kmax
                m[k] ~ Normal(1.0, 1.0)
            end
        end
        model = imm(randn(100), 1.0)
        # https://github.com/TuringLang/Turing.jl/issues/1725
        # sample(model, Gibbs(MH(:z), HMC(0.01, 4, :m)), 100);
        sample(model, Gibbs(PG(10, :z), HMC(0.01, 4, :m; adtype=adbackend)), 100)
    end
end
