module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata
using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using Distributions: Bernoulli, Beta, Gamma, Normal, sample
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_logs, @test_throws, @testset
using Turing

@testset "SMC" begin
    @testset "constructor" begin
        s = SMC()
        @test s.resampler == AlwaysResample()

        s = SMC(0.6)
        @test s.resampler === ESSResampler(0.6)
    end

    @testset "basic model" begin
        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end
        tested = sample(normal(), SMC(), 100)
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
        @test_throws "number of observations" sample(fail_smc(), SMC(), 100)
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

        @test_logs (:warn, r"ignored") sample(normal(), SMC(), 10; discard_initial=5)
        chn = sample(normal(), SMC(), 10; discard_initial=5)
        @test size(chn, 1) == 10
        @test chn isa MCMCChains.Chains

        @test_logs (:warn, r"ignored") sample(normal(), SMC(), 10; thinning=3)
        chn2 = sample(normal(), SMC(), 10; thinning=3)
        @test size(chn2, 1) == 10
        @test chn2 isa MCMCChains.Chains

        @test_logs (:warn, r"ignored") sample(
            normal(), SMC(), 10; discard_initial=2, thinning=2
        )
        chn3 = sample(normal(), SMC(), 10; discard_initial=2, thinning=2)
        @test size(chn3, 1) == 10
        @test chn3 isa MCMCChains.Chains
    end
end

@testset "PG" begin
    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(PG(SMC(), 10))
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

        chains_pg = sample(StableRNG(468), test(), PG(SMC(), 10), 100)

        @test all(isone, chains_pg[:x])
        pg_logevidence = mean(chains_pg[:logevidence])
        @test pg_logevidence ≈ -2 * log(2) atol = 0.01
        # Should be the same for all iterations.
        @test chains_pg[:logevidence] ≈ fill(pg_logevidence, 100)
    end

    # https://github.com/TuringLang/Turing.jl/issues/1598
    @testset "reference particle" begin
        c = sample(gdemo_default, PG(SMC(), 1), 1_000)
        @test length(unique(c[:m])) == 1
        @test length(unique(c[:s])) == 1
    end

    @testset "addlogprob leads to reweighting" begin
        # Make sure that PG takes @addlogprob! into account. It didn't use to:
        # https://github.com/TuringLang/Turing.jl/issues/1996
        @model function addlogprob_demo()
            x ~ Normal(0, 1)
            if x < 0
                @addlogprob! -10.0
            else
                # Need a balanced number of addlogprobs in all branches, or
                # else PG will error
                @addlogprob! 0.0
            end
        end
        c = sample(StableRNG(468), addlogprob_demo(), PG(SMC(), 10), 100)
        # Result should be biased towards x > 0.
        @test mean(c[:x]) > 0.7
    end

    @testset "keyword argument handling" begin
        @model function kwarg_demo(y; n=0.0)
            x ~ Normal(n)
            return y ~ Normal(x)
        end

        chain = sample(StableRNG(468), kwarg_demo(5.0), PG(SMC(), 20), 1000)
        @test chain isa MCMCChains.Chains
        @test mean(chain[:x]) ≈ 2.5 atol = 0.3

        chain2 = sample(StableRNG(468), kwarg_demo(5.0; n=10.0), PG(SMC(), 20), 1000)
        @test chain2 isa MCMCChains.Chains
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
        chn = sample(StableRNG(468), m1, PG(SMC(), 10), 1000)
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
        chn1 = sample(StableRNG(468), m1, PG(SMC(), 10), 1000)
        @test mean(chn1[Symbol("a.x")]) ≈ 2.5 atol = 0.3

        @model function outer_kwarg2(n)
            return a ~ to_submodel(inner_kwarg(5.0; n=n))
        end
        m2 = outer_kwarg2(10.0)
        chn2 = sample(StableRNG(468), m2, PG(SMC(), 10), 1000)
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
        @test_throws ArgumentError sample(model, PG(SMC(), 10), 100)
    end
end

end
