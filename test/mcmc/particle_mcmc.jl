module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata
using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using Distributions: Bernoulli, Beta, Gamma, Normal, sample
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
using Turing

@testset "SMC" begin
    @testset "constructor" begin
        s = SMC()
        @test s.resampler == ResampleWithESSThreshold()

        s = SMC(0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)

        s = SMC(resample_multinomial, 0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)

        s = SMC(resample_systematic)
        @test s.resampler === resample_systematic
    end

    @testset "models" begin
        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        tested = sample(normal(), SMC(), 100)

        # TODO(mhauru) This needs an explanation for why it fails.
        # failing test
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
    end

    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(SMC())
    end

    @testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains_smc = sample(test(), SMC(), 100)

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
end

@testset "PG" begin
    @testset "constructor" begin
        s = PG(10)
        @test s.nparticles == 10
        @test s.resampler == ResampleWithESSThreshold()

        s = PG(60, 0.6)
        @test s.nparticles == 60
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)

        s = PG(80, resample_multinomial, 0.6)
        @test s.nparticles == 80
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)

        s = PG(100, resample_systematic)
        @test s.nparticles == 100
        @test s.resampler === resample_systematic
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
        c = sample(StableRNG(468), addlogprob_demo(), PG(10), 100)
        # Result should be biased towards x > 0.
        @test mean(c[:x]) > 0.7
    end

    # https://github.com/TuringLang/Turing.jl/issues/2007
    @testset "keyword argument handling" begin
        @model function kwarg_demo(y; n=0.0)
            x ~ Normal(n)
            return y ~ Normal(x)
        end
        @test_throws "Models with keyword arguments" sample(kwarg_demo(5.0), PG(20), 10)

        # Check that enabling `might_produce` does allow sampling
        @might_produce kwarg_demo
        chain = sample(StableRNG(468), kwarg_demo(5.0), PG(20), 1000)
        @test chain isa MCMCChains.Chains
        @test mean(chain[:x]) ≈ 2.5 atol = 0.2

        # Check that the keyword argument's value is respected
        chain2 = sample(StableRNG(468), kwarg_demo(5.0; n=10.0), PG(20), 1000)
        @test chain2 isa MCMCChains.Chains
        @test mean(chain2[:x]) ≈ 7.5 atol = 0.2
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
