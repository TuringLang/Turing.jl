module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata
using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using Distributions: Bernoulli, Beta, Categorical, Gamma, Normal, sample
using FlexiChains: VNChain
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_logs, @test_throws, @testset
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
        @test chn isa VNChain

        @test_logs (:warn, r"ignored") sample(normal(), SMC(), 10; thinning=3)
        chn2 = sample(normal(), SMC(), 10; thinning=3)
        @test size(chn2, 1) == 10
        @test chn2 isa VNChain

        @test_logs (:warn, r"ignored") sample(
            normal(), SMC(), 10; discard_initial=2, thinning=2
        )
        chn3 = sample(normal(), SMC(), 10; discard_initial=2, thinning=2)
        @test size(chn3, 1) == 10
        @test chn3 isa VNChain
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

    @testset "conditional sweeps target the exact posterior" begin
        # Conditional SMC is invariant only if the reference is exactly the retained path, and
        # this model is shaped so that either way of getting that wrong biases the marginals.
        # The observations are sharp enough to keep the weights uneven, so resampling fires
        # and a descendant of the reference that copies it rather than branching off shows up.
        # The stay probability is the other Gibbs component, so a reference rebuilt by
        # replaying random numbers -- `z[t] = z[t-1]` exactly when `u[t] < p` -- lands on a
        # different path as soon as `i` moves. Mean absolute error over the marginals, across
        # four seeds: under 0.005 for a correct sweep, 0.020 to 0.028 for the first failure,
        # 0.017 to 0.022 for the second.
        #
        # All `2 * 2^8` configurations enumerate the exact posterior, weighted by the model's
        # own log density rather than by a reimplementation of it.
        means, sd, stay = (-1.0, 1.0), 0.8, (0.35, 0.65)
        @model function switching(y)
            i ~ Categorical(2)
            p = stay[i]
            transition = [p 1-p; 1-p p]
            z = Vector{Int}(undef, length(y))
            z[1] ~ Categorical([0.5, 0.5])
            y[1] ~ Normal(means[z[1]], sd)
            for t in 2:length(y)
                z[t] ~ Categorical(transition[z[t - 1], :])
                y[t] ~ Normal(means[z[t]], sd)
            end
        end
        y = [-0.9163, -2.4106, -2.1881, 0.3716, 1.3404, -1.2046, -1.8294, -0.3521]
        model = switching(y)
        T = length(y)

        paths = vec([collect(z) for z in Iterators.product(fill(1:2, T)...)])
        logws = [logjoint(model, (; i=i, z=path)) for i in 1:2, path in paths]
        ws = exp.(logws .- maximum(logws))
        path_probs = vec(sum(ws; dims=1)) ./ sum(ws)   # posterior over paths, `i` summed out
        exact = [path_probs' * [path[t] == 2 for path in paths] for t in 1:T]

        alg = Gibbs(@varname(i) => MH(), @varname(z) => CSMC(8))
        chn = sample(StableRNG(468), model, alg, 6_000)
        draws = stack(collect(z) for z in chn[@varname(z)])   # T x ndraws
        marginals = [mean(==(2), view(draws, t, :)) for t in 1:T]
        @test mean(abs, marginals .- exact) < 0.01
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
