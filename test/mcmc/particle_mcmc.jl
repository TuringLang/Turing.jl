module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata
using ..NumericalTests: check_numerical
using Distributions: Bernoulli, Beta, Gamma, Normal, sample
using DynamicPPL: extract_priors
using FlexiChains: VNChain
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_logs, @test_throws, @testset
using Turing
using Turing.Inference: Systematic, ESSResampler, Multinomial, smcsample, get_varinfo

@testset "SMC" begin
    @testset "constructor" begin
        s = SMC()
        @test s.resampler == Systematic()

        s = SMC(0.6)
        @test s.resampler === ESSResampler(0.6, Systematic())

        s = SMC(Multinomial(), 0.6)
        @test s.resampler === ESSResampler(0.6, Multinomial())
    end

    @testset "basic model with ensemble reweighting" begin
        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        chn1 = sample(StableRNG(23), normal(), SMC(), 10; ensemble=MCMCSerial())
        chn2 = sample(StableRNG(23), normal(), SMC(), 10; ensemble=MCMCThreads())
        chn3 = sample(StableRNG(23), normal(), SMC(), 10; ensemble=MCMCDistributed())

        # test that RNG is consistent even when parallelized
        @test chn1[@varname(a)] ≈ chn2[@varname(a)]
        @test chn1[@varname(a)] ≈ chn3[@varname(a)]
    end

    @testset "resampling schemes" begin
        @model function coinflip(y)
            p ~ Beta(1, 1)
            for t in eachindex(y)
                y[t] ~ Bernoulli(p)
            end
        end

        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        coin_model = coinflip(obs)

        prior = extract_priors(coin_model)[@varname(p)]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
        meanp = exact.α / (exact.α + exact.β)

        chn1 = sample(StableRNG(23), coin_model, SMC(Systematic()), 100)
        check_numerical(chn1, [@varname(p)], [meanp]; atol=0.1)

        chn2 = sample(StableRNG(23), coin_model, SMC(Multinomial()), 100)
        check_numerical(chn2, [@varname(p)], [meanp]; atol=0.1)

        @test chn1[@varname(p)] != chn2[@varname(p)]
    end

    @testset "errors when number of observations is not fixed" begin
        @model function fail_smc()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            if a >= 4.0
                1.5 ~ Normal(b, 2)
            end
            return a, b
        end
        @test_throws ErrorException sample(fail_smc(), SMC(), 100)
        @test_throws "mis-aligned execution" sample(fail_smc(), SMC(), 100)
    end

    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(SMC())
    end

    @testset "logevidence" begin
        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains_smc = sample(StableRNG(100), test(), SMC(), 100)

        @test all(isone, chains_smc[:x])
        # For SMC, the chain stores the collective logevidence of the sampled trajectories
        # as a statistic (which is the same for all 'iterations'). So we can just pick the
        # first one.
        smc_logevidence = first(chains_smc[:logevidence])
        @test smc_logevidence ≈ -2 * log(2)
        # Check that they're all equal.
        @test chains_smc[:logevidence] ≈ fill(smc_logevidence, 100)
    end

    @testset "refuses to run threadsafe eval" begin
        # SMC can't run models that have nondeterministic evaluation order,
        # so it should refuse to run models marked as threadsafe.
        @model function f(y)
            x ~ Normal()
            Threads.@threads for i in eachindex(y)
                y[i] ~ Normal(x)
            end
        end
        model = setthreadsafe(f(randn(10)), true)
        @test_throws ArgumentError sample(model, SMC(), 100)
    end

    @testset "discard_initial and thinning are ignored" begin
        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        chn = sample(normal(), SMC(), 10; discard_initial=5)
        @test size(chn, 1) == 10
        @test chn isa VNChain

        chn2 = sample(normal(), SMC(), 10; thinning=3)
        @test size(chn2, 1) == 10
        @test chn2 isa VNChain

        chn3 = sample(normal(), SMC(), 10; discard_initial=2, thinning=2)
        @test size(chn3, 1) == 10
        @test chn3 isa VNChain
    end
end

@testset "PG" begin
    @testset "constructor" begin
        s = PG(10)
        @test s.N == 10
        @test s.kernel.resampler === ESSResampler(0.5)

        s = PG(60, 0.6)
        @test s.N == 60
        @test s.kernel.resampler === ESSResampler(0.6)

        s = PG(80, Multinomial(), 0.6)
        @test s.N == 80
        @test s.kernel.resampler === ESSResampler(0.6, Multinomial())

        s = PG(100, Systematic())
        @test s.N == 100
        @test s.kernel.resampler === Systematic()
    end

    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(PG(10))
    end

    @testset "logevidence" begin
        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains_pg = sample(StableRNG(468), test(), PG(10), 100)

        @test all(isone, chains_pg[:x])
        pg_logevidence = mean(chains_pg[:logevidence])
        @test pg_logevidence ≈ -2 * log(2) atol = 0.01
        # Should be the same for all iterations.
        @test chains_pg[:logevidence] ≈ fill(pg_logevidence, 100)
    end

    # https://github.com/TuringLang/Turing.jl/issues/1598
    @testset "reference particle" begin
        c = sample(gdemo_default, PG(1), 1_000)
        @test length(unique(c[:m])) == 1
        @test length(unique(c[:s])) == 1
    end

    @testset "ensuring reference consistency" begin
        # NOTE: this test fails when the RNG counter doesn't align with the retained keys
        get_raw_vals(trace::TracedModel) = DynamicPPL.get_raw_values(get_varinfo(trace))

        @model function state_space_model(y)
            ρ ~ Uniform(0, 1)
            x = Vector{Float64}(undef, length(y) + 1)
            x[1] ~ Normal(0, 1)
            for t in eachindex(y)
                x[t + 1] ~ Normal(ρ * x[t], 1)
                y[t] ~ Normal(x[t + 1], 1)
            end
        end

        rng = StableRNG(1234)
        y = randn(rng, 10)
        ss_model = state_space_model(y)

        particles, logZ = smcsample(rng, ss_model, SMC(0.5), MCMCSerial(), 3)
        state = sample(rng, particles)
        check = Vector{Bool}(undef, 30)

        for m in eachindex(check)
            ref = deepcopy(state)
            particles, _ = smcsample(rng, ss_model, SMC(0.5), MCMCSerial(), 3; ref)
            check[m] = (get_raw_vals(particles.reference.value) == get_raw_vals(state))
            state = sample(rng, particles)
        end

        # ensure reference is perfectly regenerated
        @test all(check)
        @test length(Turing.Inference.get_rng(state).keys) == (length(y) + 1)
    end

    # https://github.com/TuringLang/Turing.jl/issues/1996
    @testset "addlogprob leads to reweighting" begin
        # Make sure that PG takes @addlogprob! into account
        @model function addlogprob_demo()
            x ~ Normal(0, 1)
            if x < 0
                @producelogprob! -10.0
            else
                # Need a balanced number of addlogprobs in all branches, or
                # else PG will error
                @producelogprob! 0.0
            end
        end
        c = sample(StableRNG(468), addlogprob_demo(), PG(10), 100)
        # Result should be biased towards x > 0.
        @test mean(c[:x]) > 0.7
    end

    @testset "keyword argument handling" begin
        @model function kwarg_demo(y; n=0.0)
            x ~ Normal(n)
            return y ~ Normal(x)
        end

        chain = sample(StableRNG(468), kwarg_demo(5.0), PG(20), 1000)
        @test chain isa VNChain
        @test mean(chain[:x]) ≈ 2.5 atol = 0.3

        chain2 = sample(StableRNG(468), kwarg_demo(5.0; n=10.0), PG(20), 1000)
        @test chain2 isa VNChain
        @test mean(chain2[:x]) ≈ 7.5 atol = 0.3
    end

    @testset "submodels without kwargs" begin
        @model function inner(y, x)
            # Mark as noinline explicitly to make sure that behaviour is not reliant on the
            # Julia compiler inlining it.
            # See https://github.com/TuringLang/Turing.jl/issues/2772
            @noinline
            return y ~ Normal(x)
        end
        @model function nested(y)
            x ~ Normal()
            return a ~ to_submodel(inner(y, x))
        end
        m1 = nested(1.0)
        chn = sample(StableRNG(468), m1, PG(10), 1000)
        @test mean(chn[:x]) ≈ 0.5 atol = 0.1
    end

    @testset "submodels with kwargs" begin
        @model function inner_kwarg(y; n=0.0)
            @noinline # See above
            x ~ Normal(n)
            return y ~ Normal(x)
        end
        @model function outer_kwarg1()
            return a ~ to_submodel(inner_kwarg(5.0))
        end
        m1 = outer_kwarg1()
        chn1 = sample(StableRNG(468), m1, PG(10), 1000)
        @test mean(chn1[Symbol("a.x")]) ≈ 2.5 atol = 0.3

        @model function outer_kwarg2(n)
            return a ~ to_submodel(inner_kwarg(5.0; n=n))
        end
        m2 = outer_kwarg2(10.0)
        chn2 = sample(StableRNG(468), m2, PG(10), 1000)
        @test mean(chn2[Symbol("a.x")]) ≈ 7.5 atol = 0.3
    end

    @testset "refuses to run threadsafe eval" begin
        # PG can't run models that have nondeterministic evaluation order,
        # so it should refuse to run models marked as threadsafe.
        @model function f(y)
            x ~ Normal()
            Threads.@threads for i in eachindex(y)
                y[i] ~ Normal(x)
            end
        end
        model = setthreadsafe(f(randn(10)), true)
        @test_throws ArgumentError sample(model, PG(10), 100)
    end
end

end
