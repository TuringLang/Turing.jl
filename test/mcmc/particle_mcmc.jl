module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata, test_rng_respected
using ..NumericalTests: check_numerical
using DynamicPPL: DynamicPPL, extract_priors, get_raw_values, getloglikelihood
using Turing.Inference:
    Stratified,
    Systematic,
    Multinomial,
    ESSResampler,
    Particle,
    TracedRNG,
    particle_varinfo,
    advance!,
    fork,
    rewind!,
    refresh!,
    sweep!,
    normalized_weights,
    PGState
using Distributions: Bernoulli, Beta, Gamma, Normal, Uniform, Categorical, sample
using FlexiChains: VNChain, has_same_data
using Random: Random, Xoshiro
using StableRNGs: StableRNG
using Test: @test, @test_logs, @test_throws, @testset
using Turing

@testset "SMC" begin
    @testset "constructor" begin
        @test SMC().resampler == ESSResampler(0.5)
        @test SMC().resampler.scheme isa Stratified   # stratified is the default scheme
        @test SMC(0.6).resampler == ESSResampler(0.6)
        @test SMC(Multinomial(), 0.6).resampler == ESSResampler(0.6, Multinomial())
        @test SMC(Systematic()).resampler == Systematic()
        @test SMC().threaded == false
        @test SMC(; threaded=true).threaded == true
        @test SMC(Systematic(); threaded=true).threaded == true
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

        # every scheme targets the same posterior...
        chn_strat = sample(StableRNG(23), coin_model, SMC(Stratified()), 100)
        chn_multi = sample(StableRNG(23), coin_model, SMC(Multinomial()), 100)
        check_numerical(chn_strat, [@varname(p)], [mean(exact)]; atol=0.1)
        check_numerical(chn_multi, [@varname(p)], [mean(exact)]; atol=0.1)
        # ...but the schemes are genuinely different, so the draws differ.
        @test chn_strat[@varname(p)] != chn_multi[@varname(p)]
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

    @testset "rng is respected" begin
        test_rng_respected(SMC())
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

    @testset "threaded execution matches serial" begin
        # Particles are seeded serially before the parallel reweighting, so `threaded=true`
        # must reproduce the serial draws exactly (bit for bit), whatever the thread count.
        @model function coinflip(y)
            p ~ Beta(1, 1)
            for t in eachindex(y)
                y[t] ~ Bernoulli(p)
            end
        end
        model = coinflip([0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
        serial = sample(StableRNG(23), model, SMC(), 200)
        threaded = sample(StableRNG(23), model, SMC(; threaded=true), 200)
        @test serial[@varname(p)] == threaded[@varname(p)]
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
        @test PG(10).nparticles == 10
        @test PG(10).resampler == ESSResampler(0.5)
        @test PG(60, 0.6).resampler == ESSResampler(0.6)
        @test PG(80, Multinomial(), 0.6).resampler == ESSResampler(0.6, Multinomial())
        @test PG(100, Systematic()).resampler == Systematic()
        @test PG(10).threaded == false
        @test PG(10; threaded=true).threaded == true
        @test PG(80, Multinomial(), 0.6; threaded=true).threaded == true
    end

    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(PG(10))
    end

    @testset "rng is respected" begin
        test_rng_respected(PG(10))
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

    @testset "threaded execution matches serial" begin
        # Threading the reweighting must not perturb the reference-replay bookkeeping, so the
        # conditional sweeps have to reproduce the serial draws exactly, whatever the thread
        # count.
        @model function coinflip(y)
            p ~ Beta(1, 1)
            for t in eachindex(y)
                y[t] ~ Bernoulli(p)
            end
        end
        model = coinflip([0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
        serial = sample(StableRNG(23), model, PG(10), 200)
        threaded = sample(StableRNG(23), model, PG(10; threaded=true), 200)
        @test serial[@varname(p)] == threaded[@varname(p)]
    end

    # https://github.com/TuringLang/Turing.jl/issues/1598
    @testset "reference particle" begin
        c = sample(gdemo_default, PG(1), 1_000)
        @test length(unique(c[:m])) == 1
        @test length(unique(c[:s])) == 1
    end

    @testset "ensuring reference consistency" begin
        # In conditional SMC the retained trajectory must be regenerated *exactly* by the
        # reference particle on the next iteration -- this is what makes CSMC valid. It fails
        # if the traced-RNG step counter and the recorded seeds ever fall out of alignment.
        @model function state_space_model(y)
            ρ ~ Uniform(0, 1)
            x = Vector{Float64}(undef, length(y) + 1)
            x[1] ~ Normal(0, 1)
            for t in eachindex(y)
                x[t + 1] ~ Normal(ρ * x[t], 1)
                y[t] ~ Normal(x[t + 1], 1)
            end
        end

        # Run PG's conditional sweep by hand so we can inspect the reference particle (slot
        # N) and check it reproduces the trajectory we retained. Wrapped in a function to keep
        # the mutating loop out of test soft scope.
        function run_csmc(model, N, nsteps, rng)
            draw(ps) = ps[rand(rng, Categorical(normalized_weights(ps)))]
            particles = [Particle(model, particle_varinfo(), TracedRNG(rng)) for _ in 1:N]
            sweep!(rng, particles, ESSResampler(0.5), false)
            state = let p = draw(particles)
                PGState(p.varinfo, p.rng)
            end
            allok = true
            for _ in 1:nsteps
                ref = Particle(model, particle_varinfo(), rewind!(deepcopy(state.rng)))
                parts = map(
                    i -> i < N ? Particle(model, particle_varinfo(), TracedRNG(rng)) : ref,
                    1:N,
                )
                sweep!(rng, parts, ESSResampler(0.5), false; conditional=true)
                allok &= get_raw_values(parts[N].varinfo) == get_raw_values(state.varinfo)
                p = draw(parts)
                state = PGState(p.varinfo, p.rng)
            end
            return allok, length(state.rng.keys)
        end

        rng = StableRNG(1234)
        y = randn(rng, 10)
        allok, nkeys = run_csmc(state_space_model(y), 3, 30, rng)
        @test allok                         # reference regenerated exactly every step
        @test nkeys == length(y) + 1        # keys stay aligned with the trajectory length
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

        # @addlogprob! should also be respected by ordinary (non-particle) samplers.
        c2 = sample(StableRNG(468), addlogprob_demo(), MH(), 100)
        @test mean(c2[:x]) > 0.7
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

@testset "parallel chains (MCMCThreads)" begin
    @model function coinflip(y)
        p ~ Beta(1, 1)
        for t in eachindex(y)
            y[t] ~ Bernoulli(p)
        end
    end
    model = coinflip([0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
    # Multiple chains through AbstractMCMC's thread-based ensemble stay reproducible under a
    # fixed seed (genuinely parallel only when Julia is started with more than one thread).
    for sampler in (SMC(), PG(10))
        c1 = sample(Xoshiro(5), model, sampler, MCMCThreads(), 100, 4)
        c2 = sample(Xoshiro(5), model, sampler, MCMCThreads(), 100, 4)
        @test has_same_data(c1, c2)
    end
end

@testset "particle container" begin
    @model function test()
        a ~ Normal(0, 1)
        x ~ Bernoulli(1)
        b ~ Gamma(2, 3)
        1 ~ Bernoulli(x / 2)
        c ~ Beta()
        0 ~ Bernoulli(x / 2)
        return x
    end

    @testset "advance!" begin
        # `x ~ Bernoulli(1)` forces `x = 1`, so the first observe is `1 ~ Bernoulli(0.5)`.
        particle = Particle(test(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        @test advance!(particle, false) ≈ -log(2)
        @test advance!(particle, false) ≈ -log(2)     # `0 ~ Bernoulli(0.5)`
        @test advance!(particle, false) === nothing    # model finished
    end

    @testset "matches a direct evaluation" begin
        # A particle advanced without resampling draws from its RNG continuously, so it must
        # produce exactly the same values and log-likelihood as a plain DynamicPPL evaluation
        # seeded identically.
        particle = Particle(test(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        while advance!(particle, false) !== nothing
        end

        accs = DynamicPPL.OnlyAccsVarInfo()
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.LogLikelihoodAccumulator())
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(true))
        _, accs = DynamicPPL.init!!(
            TracedRNG(Xoshiro(23)),
            test(),
            accs,
            DynamicPPL.InitFromPrior(),
            DynamicPPL.UnlinkAll(),
        )

        @test get_raw_values(particle.varinfo) == get_raw_values(accs)
        @test getloglikelihood(particle.varinfo) == getloglikelihood(accs)
    end

    @testset "fork" begin
        particle = Particle(test(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        advance!(particle, false)
        child = fork(particle, Xoshiro(1))
        # Independent continuations: advancing one does not touch the other.
        @test advance!(child, false) ≈ -log(2)
        @test particle.varinfo !== child.varinfo
        @test advance!(particle, false) ≈ -log(2)
    end

    @testset "rng replay" begin
        @model function normal()
            a ~ Normal(0, 1)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        # Run a particle to completion, then replay it from its recorded seeds (as the
        # reference trajectory of a conditional sweep does) and check it regenerates exactly.
        # Replay relies on each step using a distinct seed, so we refresh before every step
        # exactly as the sweep's no-resample path does.
        particle = Particle(normal(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        while (refresh!(particle.rng); advance!(particle, false)) !== nothing
        end
        values = DynamicPPL.get_raw_values(particle.varinfo)

        reference = Particle(normal(), particle_varinfo(), rewind!(deepcopy(particle.rng)))
        while advance!(reference, true) !== nothing
        end
        @test DynamicPPL.get_raw_values(reference.varinfo) == values
    end
end

end
