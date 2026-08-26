module ParticleMCMCTests

using ..Models: gdemo_default
using ..SamplerTestUtils: test_chain_logp_metadata, test_rng_respected
using ..NumericalTests: check_numerical
using ..ExactSSM: ExactSSM
using DynamicPPL: DynamicPPL, extract_priors, get_raw_values, getloglikelihood
using Turing.Inference:
    StratifiedResampler,
    SystematicResampler,
    MultinomialResampler,
    ESSThresholdResampler,
    Particle,
    particle_rng,
    advance!,
    fork,
    sweep!,
    resample_indices,
    pg_transition_and_state
using Distributions:
    Bernoulli,
    Beta,
    Categorical,
    Exponential,
    Gamma,
    InverseGamma,
    LogNormal,
    MvNormal,
    Normal,
    Poisson,
    Uniform,
    logpdf,
    product_distribution,
    sample
using FlexiChains: VNChain, has_same_data
using LinearAlgebra: I
using Random: Random, Xoshiro
using Serialization: deserialize, serialize
using SpecialFunctions: logbeta
using StableRNGs: StableRNG
using Test: @test, @test_logs, @test_throws, @testset
using Turing

#
# Shared models
#

# Models shared across the testsets below. Defined at module scope, not inside a testset: a `@model`
# sharing a local scope with a same-named variable captures it, and every particle then mutates one
# shared array (see the `xtrue`/`ztrue` note further down).

@model function coinflip(y)
    p ~ Beta(1, 1)
    for t in eachindex(y)
        y[t] ~ Bernoulli(p)
    end
end
const COIN_OBS = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

# `x ~ Bernoulli(1)` pins x = 1, so both observes contribute exactly log(1/2) whatever the
# trajectory -- zero weight variance, which several log_normalizing_constant tests rely on.
@model function test()
    a ~ Normal(0, 1)
    x ~ Bernoulli(1)
    b ~ Gamma(2, 3)
    1 ~ Bernoulli(x / 2)
    c ~ Beta()
    0 ~ Bernoulli(x / 2)
    return x
end

@model function normal()
    a ~ Normal(4, 5)
    3 ~ Normal(a, 2)
    b ~ Normal(a, 1)
    1.5 ~ Normal(b, 2)
    return a, b
end

# As `normal()` but centred at zero; used where the replay test wants a different trajectory.
@model function centred_normal()
    a ~ Normal(0, 1)
    3 ~ Normal(a, 2)
    b ~ Normal(a, 1)
    1.5 ~ Normal(b, 2)
    return a, b
end

# Nondeterministic evaluation order, which the particle samplers must refuse.
@model function threadsafe_model(y)
    x ~ Normal()
    Threads.@threads for i in eachindex(y)
        y[i] ~ Normal(x)
    end
end

"Run a particle to completion."
run_to_end!(p) = (while advance!(p) !== nothing
end;
p)

#
# SMC
#

@testset "SMC" begin
    @testset "constructor" begin
        @test SMC().resampler == ESSThresholdResampler(0.5)
        @test SMC().resampler.scheme isa StratifiedResampler   # stratified is the default scheme
        @test SMC(0.6).resampler == ESSThresholdResampler(0.6)
        @test SMC(MultinomialResampler(), 0.6).resampler ==
            ESSThresholdResampler(0.6, MultinomialResampler())
        @test SMC(SystematicResampler()).resampler == SystematicResampler()
        @test SMC().multithreaded == false
        @test SMC(; multithreaded=true).multithreaded == true
        @test SMC(SystematicResampler(); multithreaded=true).multithreaded == true
    end

    @testset "basic model" begin
        tested = sample(normal(), SMC(), 100)
    end

    @testset "resampling schemes" begin
        obs = COIN_OBS
        coin_model = coinflip(obs)
        prior = extract_priors(coin_model)[@varname(p)]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))

        # every scheme targets the same posterior...
        chn_strat = sample(StableRNG(23), coin_model, SMC(StratifiedResampler()), 100)
        chn_multi = sample(StableRNG(23), coin_model, SMC(MultinomialResampler()), 100)
        check_numerical(chn_strat, [@varname(p)], [mean(exact)]; atol=0.1)
        check_numerical(chn_multi, [@varname(p)], [mean(exact)]; atol=0.1)
        # ...but the schemes are genuinely different, so the draws differ.
        @test chn_strat[@varname(p)] != chn_multi[@varname(p)]
    end

    @testset "stratified/systematic resampling never index past the end" begin
        # softmax can return weights summing to slightly under one; the cumulative walk must
        # not run off the end of the last stratum. Exaggerate the undersum so the (otherwise
        # astronomically rare) overrun is hit on every seed.
        weights = fill(0.9 / 8, 8)
        for scheme in (StratifiedResampler(), SystematicResampler())
            @test all(
                all(in(1:8), resample_indices(Xoshiro(s), scheme, weights, 8)) for
                s in 1:1000
            )
        end
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

    @testset "log_normalizing_constant" begin
        chains_smc = sample(StableRNG(100), test(), SMC(), 100)

        @test all(isone, chains_smc[:x])
        # For SMC, the chain stores the collective log_normalizing_constant of the sampled trajectories
        # as a statistic (which is the same for all 'iterations'). So we can just pick the
        # first one.
        smc_log_normalizing_constant = first(chains_smc[:log_normalizing_constant])
        @test smc_log_normalizing_constant ≈ -2 * log(2)
        # Check that they're all equal.
        @test chains_smc[:log_normalizing_constant] ≈
            fill(smc_log_normalizing_constant, 100)
    end

    @testset "multithreaded execution matches serial" begin
        # Particles are seeded serially before the parallel reweighting, so `multithreaded=true`
        # must reproduce the serial draws exactly (bit for bit), whatever the thread count.
        model = coinflip(COIN_OBS)
        serial = sample(StableRNG(23), model, SMC(), 200)
        multithreaded = sample(StableRNG(23), model, SMC(; multithreaded=true), 200)
        @test serial[@varname(p)] == multithreaded[@varname(p)]
    end

    @testset "refuses to run threadsafe eval" begin
        # SMC can't run models that have nondeterministic evaluation order,
        # so it should refuse to run models marked as threadsafe.
        model = setthreadsafe(threadsafe_model(randn(10)), true)
        @test_throws ArgumentError sample(model, SMC(), 100)
    end

    @testset "discard_initial, thinning, initial_params and callback are ignored" begin
        @test_logs (:warn, r"initial_params.*ignored") match_mode = :any sample(
            normal(), SMC(), 10; initial_params=(; a=1.0)
        )
        @test_logs (:warn, r"initial_params.*ignored") sample(
            normal(), SMC(), 10; initial_params=DynamicPPL.InitFromUniform()
        )

        # The ensemble wrapper injects the sampler's own default `InitFromPrior()` per chain.
        # That is not a user-specified initialisation, so it must not warn.
        @test_logs sample(Xoshiro(1), normal(), SMC(), MCMCSerial(), 10, 2; progress=false)

        # A callback is accepted only to be reported as ignored, and must not run.
        called = false
        @test_logs (:warn, r"callback.*ignored") sample(
            normal(), SMC(), 10; callback=(args...; kwargs...) -> (called = true)
        )
        @test !called

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

#
# PG / conditional SMC
#

@testset "PG" begin
    @testset "constructor" begin
        @test PG(10).nparticles == 10
        @test PG(10).resampler == ESSThresholdResampler(0.5)
        @test PG(60, 0.6).resampler == ESSThresholdResampler(0.6)
        @test PG(80, MultinomialResampler(), 0.6).resampler ==
            ESSThresholdResampler(0.6, MultinomialResampler())
        @test PG(100, SystematicResampler()).resampler == SystematicResampler()
        @test PG(10).multithreaded == false
        @test PG(10; multithreaded=true).multithreaded == true
        @test PG(80, MultinomialResampler(), 0.6; multithreaded=true).multithreaded == true
    end

    @testset "chain log-density metadata" begin
        test_chain_logp_metadata(PG(10))
    end

    @testset "rng is respected" begin
        test_rng_respected(PG(10))
    end

    @testset "log_normalizing_constant" begin
        chains_pg = sample(StableRNG(468), test(), PG(10), 100)

        @test all(isone, chains_pg[:x])
        pg_log_normalizing_constant = mean(chains_pg[:log_normalizing_constant])
        @test pg_log_normalizing_constant ≈ -2 * log(2) atol = 0.01
        # Every particle scores the same here -- `x ~ Bernoulli(1)` pins `x = 1`, so both observes
        # contribute exactly `log(1/2)` regardless of the trajectory. Zero weight variance is why
        # the estimate is exact for PG too, and why all iterations agree. It is *not* evidence that
        # PG's estimator is unbiased in general; the testset below covers that.
        @test chains_pg[:log_normalizing_constant] ≈ fill(pg_log_normalizing_constant, 100)
    end

    @testset "log_normalizing_constant is biased upward for conditional sweeps" begin
        # Unlike SMC's, PG's `log_normalizing_constant` is not an unbiased estimate of log p(y):
        # a conditional sweep keeps the reference whatever its weight, and the reference is a
        # posterior draw rather than a proposal draw, so it inflates the mean weight at each step.
        # Pin the direction and rough size so a future change to the sweep cannot quietly alter it.
        #
        # Beta-Bernoulli, so p(y) is exact: with a Beta(1,1) prior, p(y) = B(1+s, 1+n-s)/B(1,1).
        obs = COIN_OBS
        s, n = sum(obs), length(obs)
        exact_logp = logbeta(1 + s, 1 + n - s) - logbeta(1, 1)

        # SMC's estimate is unbiased, so averaging Ẑ over independent sweeps lands on p(y).
        smc_ratios = map(1:200) do i
            chn = sample(StableRNG(900 + i), coinflip(obs), SMC(), 32)
            exp(first(chn[:log_normalizing_constant]) - exact_logp)
        end
        @test mean(smc_ratios) ≈ 1 atol = 0.1

        # PG's overshoots. Drop iteration 1, which is an unconditional sweep.
        chn = sample(StableRNG(468), coinflip(obs), PG(8), 2_000)
        pg_ratios = exp.(vec(collect(chn[:log_normalizing_constant]))[2:end] .- exact_logp)
        @test mean(pg_ratios) > 1.05
    end

    @testset "multithreaded execution matches serial" begin
        # Threading the reweighting must not perturb the reference-replay bookkeeping, so the
        # conditional sweeps have to reproduce the serial draws exactly, whatever the thread
        # count.
        model = coinflip(COIN_OBS)
        serial = sample(StableRNG(23), model, PG(10), 200)
        multithreaded = sample(StableRNG(23), model, PG(10; multithreaded=true), 200)
        @test serial[@varname(p)] == multithreaded[@varname(p)]
    end

    @testset "conditional sweeps ignore the named resampling scheme" begin
        # Conditional sweeps draw their ancestors multinomially whatever scheme is named, so
        # sweeps differing only in that scheme must agree exactly. Bare schemes (no ESS gate)
        # resample at every step, so the draw is exercised throughout.
        @model function drifting(y)
            x ~ Normal()
            for t in eachindex(y)
                y[t] ~ Normal(x, 1)
            end
        end
        model = drifting([0.3, -0.7, 1.1])
        function conditional_sweep(scheme)
            rng = StableRNG(77)
            retained = Particle(model, particle_rng(rng))
            run_to_end!(retained)
            reference = Particle(model, particle_rng(rng), get_raw_values(retained.varinfo))
            particles = [Particle(model, particle_rng(rng)) for _ in 1:4]
            push!(particles, reference)
            sweep!(StableRNG(78), particles, scheme, false)
            return map(p -> get_raw_values(p.varinfo), particles)
        end
        multinomial = conditional_sweep(MultinomialResampler())
        @test conditional_sweep(StratifiedResampler()) == multinomial
        @test conditional_sweep(SystematicResampler()) == multinomial
    end

    # https://github.com/TuringLang/Turing.jl/issues/1598
    @testset "reference particle" begin
        c = sample(gdemo_default, PG(1), 1_000)
        @test length(unique(c[:m])) == 1
        @test length(unique(c[:s])) == 1
    end

    @testset "initial_params is ignored" begin
        # PG's first sweep draws from the prior, so there is nowhere to put a starting point.
        @test_logs (:warn, r"initial_params.*ignored") match_mode = :any sample(
            normal(), PG(5), 10; initial_params=(; a=1.0)
        )
        # `InitFromPrior()` is what the ensemble wrapper injects per chain, not a user request.
        @test_logs sample(Xoshiro(1), normal(), PG(5), MCMCSerial(), 10, 2; progress=false)
    end

    @testset "the saved state survives serialisation" begin
        chn = sample(StableRNG(24), gdemo_default, PG(5), 10; save_state=true)
        io = IOBuffer()
        serialize(io, chn)
        seekstart(io)
        state = only(loadstate(deserialize(io)))
        # With one particle the reference is the whole population, so every draw is the retained
        # trajectory: the round-tripped state has to still carry it.
        resumed = sample(StableRNG(25), gdemo_default, PG(1), 5; initial_state=state)
        @test length(unique(resumed[:m])) == 1
    end

    @testset "ensuring reference consistency" begin
        # In conditional SMC the retained trajectory must be regenerated *exactly* by the
        # reference particle on the next iteration -- this is what makes CSMC valid. Reusing the
        # retained values must reach every latent address, over many sweeps and after the
        # reference has itself been resampled from.
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
            # The sampler's own selection rule, not a reimplementation of it: if `pg_transition_and`
            # `_state` ever changes how the retained particle is chosen, this test follows.
            draw(ps) = last(pg_transition_and_state(rng, ps, 0.0, true))
            particles = [Particle(model, particle_rng(rng)) for _ in 1:N]
            sweep!(rng, particles, ESSThresholdResampler(0.5), false)
            state = draw(particles)
            allok = true
            nretained = 0
            for _ in 1:nsteps
                ref = Particle(model, particle_rng(rng), state.trajectory)
                parts = [Particle(model, particle_rng(rng)) for _ in 1:(N - 1)]
                push!(parts, ref)
                sweep!(rng, parts, ESSThresholdResampler(0.5), false)
                allok &= get_raw_values(parts[N].varinfo) == state.trajectory
                state = draw(parts)
                nretained = length(keys(state.trajectory))
            end
            return allok, nretained
        end

        rng = StableRNG(1234)
        y = randn(rng, 10)
        # ρ plus x[1:length(y)+1]: the retained trajectory must span every latent, so that the
        # next reference is pinned on all of them rather than silently redrawing the rest.
        allok, nretained = run_csmc(state_space_model(y), 3, 30, rng)
        @test allok                             # reference regenerated exactly every step
        @test nretained == length(y) + 2
    end

    @testset "reference is pinned to retained values under re-conditioning" begin
        # In Gibbs the model is re-conditioned between sweeps, so the CSMC reference must
        # reproduce the *retained values* rather than re-draw them from the (now different) prior.
        # Retain a trajectory under one conditioning, then rebuild the reference under another:
        # value-pinning keeps the trajectory, whereas re-drawing would follow the shifted prior.
        @model function reconditioned(y)
            a ~ Normal(0, 10)
            x ~ Normal(a, 1)                # x's prior depends on a, owned by another component
            return y ~ Normal(x, 1)
        end
        rng = StableRNG(42)
        retained = Particle(reconditioned(2.0) | (@varname(a) => 0.0), particle_rng(rng))
        run_to_end!(retained)
        retained_vals = get_raw_values(retained.varinfo)
        reference = Particle(
            reconditioned(2.0) | (@varname(a) => 5.0),   # x's prior shifted far away
            particle_rng(rng),
            retained_vals,
        )
        run_to_end!(reference)
        @test get_raw_values(reference.varinfo) == retained_vals
    end

    @testset "value replay detects a changed latent trace" begin
        @model function branch_changes(flag, y)
            if flag
                x ~ Normal()
                μ = x
            else
                z ~ Normal()
                μ = z
            end
            return y ~ Normal(μ, 1)
        end
        rng = StableRNG(91)
        retained = Particle(branch_changes(true, 0.0), particle_rng(rng))
        run_to_end!(retained)
        reference = Particle(
            branch_changes(false, 0.0), particle_rng(rng), get_raw_values(retained.varinfo)
        )
        @test_throws "reference execution trace changed" advance!(reference)

        @model function branch_drops(flag, y)
            x ~ Normal()
            if flag
                z ~ Normal()
            end
            return y ~ Normal(x, 1)
        end
        retained = Particle(branch_drops(true, 0.0), particle_rng(rng))
        run_to_end!(retained)
        reference = Particle(
            branch_drops(false, 0.0), particle_rng(rng), get_raw_values(retained.varinfo)
        )
        @test_throws "reference execution trace changed" begin
            run_to_end!(reference)
        end
    end

    @testset "value replay handles a slice assume" begin
        # `x[1:2] ~ MvNormal(...)` is assumed under the single address `x[1:2]` but stored in the
        # retained values under the keys `x[1]`, `x[2]`. The reference has to reuse it anyway, which
        # it does because `haskey` resolves the slice against those keys.
        @model function slice_assume(y)
            x = Vector{Float64}(undef, 2)
            x[1:2] ~ MvNormal(zeros(2), I)
            y[1] ~ Normal(x[1], 0.5)
            return y[2] ~ Normal(x[2], 0.5)
        end
        chn = sample(StableRNG(105), slice_assume([0.4, -0.4]), PG(5), 20)
        @test size(chn, 1) == 20
    end

    @testset "latents whose dimension varies between executions" begin
        # The two testsets above cover traces that *changed* and must be rejected. This covers one
        # that is legitimately different on every execution and must simply work: `k[t]` decides how
        # many jumps step `t` has, so the reference must reuse `k[t]` before it can reuse a jump
        # vector of the matching length.
        #
        # The target is exact without a reference implementation, and still informative about the
        # varying dimension. Tilting `k[t] ~ Poisson(1)` by `c^k[t]` gives exactly `Poisson(c)`,
        # since `e⁻¹c^k/k!` normalises to `e⁻ᶜc^k/k!`. The tilt is the only term that carries
        # information, and it is a function of the trace's shape, so a reference that replayed or
        # reweighted the varying-length part wrongly would move `E[k[t]]` off `c`. The observation
        # itself ignores the latents; it is there to give the sweep its produce points.
        tilt = 2.0
        @model function random_dimension(y, c)
            k = Vector{Int}(undef, length(y))
            jumps = Vector{Vector{Float64}}(undef, length(y))
            for t in eachindex(y)
                k[t] ~ Poisson(1.0)
                if k[t] > 0
                    jumps[t] ~ product_distribution(fill(Exponential(1.0), k[t]))
                else
                    # A zero-length `product_distribution` is not usable, and `k[t] = 0` has
                    # probability e⁻¹, so this branch is taken constantly.
                    jumps[t] = Float64[]
                end
                @addlogprob! k[t] * log(c)
                y[t] ~ Normal(0.0, 1.0)
            end
        end

        ndraws = 2_000
        chn = sample(StableRNG(106), random_dimension(zeros(4), tilt), PG(16), ndraws)
        @test size(chn, 1) == ndraws
        ks = reduce(vcat, (reshape(collect(k), 1, :) for k in collect(chn[@varname(k)])))
        @test size(ks) == (ndraws, 4)
        # `sqrt(tilt / ndraws)` is the standard error this mean would have from independent draws.
        # A PG chain is autocorrelated, and measured across eight seeds its batch-means standard
        # error runs about 2.3x that, so eight iid errors is roughly 3.5 real ones. The margin is
        # there to keep the test from flaking, and it costs nothing here: the failure mode this
        # guards against -- dropping the weight that depends on the trace's shape -- lands on the
        # prior mean 1.0, four times the tolerance away.
        tol = 8 * sqrt(tilt / ndraws)
        for t in 1:4
            @test mean(@view ks[:, t]) ≈ tilt atol = tol
            @test mean(==(0), @view ks[:, t]) ≈ exp(-tilt) atol = tol
        end
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
        chn2 = sample(StableRNG(468), m2, PG(10), 2000)
        @test mean(chn2[Symbol("a.x")]) ≈ 7.5 atol = 0.3
    end

    @testset "refuses to run threadsafe eval" begin
        # PG can't run models that have nondeterministic evaluation order,
        # so it should refuse to run models marked as threadsafe.
        model = setthreadsafe(threadsafe_model(randn(10)), true)
        @test_throws ArgumentError sample(model, PG(10), 100)
    end
end

#
# Chain-level parallelism
#

@testset "parallel chains (MCMCThreads)" begin
    model = coinflip(COIN_OBS)
    # Multiple chains through AbstractMCMC's thread-based ensemble stay reproducible under a
    # fixed seed (genuinely parallel only when Julia is started with more than one thread).
    for sampler in (SMC(), PG(10))
        c1 = sample(Xoshiro(5), model, sampler, MCMCThreads(), 100, 4)
        c2 = sample(Xoshiro(5), model, sampler, MCMCThreads(), 100, 4)
        @test has_same_data(c1, c2)
    end
end

#
# Particle mechanics
#

@testset "particle container" begin
    @testset "advance!" begin
        # `x ~ Bernoulli(1)` forces `x = 1`, so the first observe is `1 ~ Bernoulli(0.5)`.
        particle = Particle(test(), particle_rng(Xoshiro(23)))
        @test advance!(particle) ≈ -log(2)
        @test advance!(particle) ≈ -log(2)     # `0 ~ Bernoulli(0.5)`
        @test advance!(particle) === nothing    # model finished
    end

    @testset "matches a direct evaluation" begin
        # A particle advanced without resampling draws from its RNG continuously, so it must
        # produce exactly the same values and log-likelihood as a plain DynamicPPL evaluation
        # seeded identically.
        particle = Particle(test(), particle_rng(Xoshiro(23)))
        run_to_end!(particle)

        accs = DynamicPPL.OnlyAccsVarInfo()
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.LogLikelihoodAccumulator())
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(true))
        _, accs = DynamicPPL.init!!(
            particle_rng(Xoshiro(23)),
            test(),
            accs,
            DynamicPPL.InitFromPrior(),
            DynamicPPL.UnlinkAll(),
        )

        @test get_raw_values(particle.varinfo) == get_raw_values(accs)
        @test getloglikelihood(particle.varinfo) == getloglikelihood(accs)
    end

    @testset "fork" begin
        particle = Particle(test(), particle_rng(Xoshiro(23)))
        advance!(particle)
        child = fork(particle, Xoshiro(1))
        # Independent continuations: advancing one does not touch the other.
        @test advance!(child) ≈ -log(2)
        @test particle.varinfo !== child.varinfo
        @test advance!(particle) ≈ -log(2)
    end

    @testset "reference consumes no randomness" begin
        # The reference reproduces the retained trajectory purely from its values, so its own
        # generator is never consulted. Pinning that down is what lets the reference be handed an
        # ordinary generator instead of a replayable one: scrambling its seeds before every step
        # must not perturb the trajectory it regenerates.
        retained = Particle(centred_normal(), particle_rng(Xoshiro(23)))
        run_to_end!(retained)
        values = get_raw_values(retained.varinfo)

        scrambler = Xoshiro(99)
        reference = Particle(normal(), particle_rng(Xoshiro(7)), values)
        while (
            Random.seed!(reference.rng, rand(scrambler, UInt64)); advance!(reference)
        ) !== nothing
        end
        @test get_raw_values(reference.varinfo) == values
    end
end

#
# State space models with tractable posteriors
#

# These are the checks that pin the particle samplers to a known answer rather than to each other. A
# scalar linear Gaussian SSM and a discrete HMM both have closed-form posteriors, supplied by
# `ExactSSM` and validated there against brute force, so a disagreement beyond Monte Carlo error is a
# bug.
#
# Each model is exercised twice: `PG` alone against the exact smoothing marginals, then
# `Gibbs(θ => NUTS/HMC, states => CSMC)` with a static parameter unknown. The second is the case that
# matters, because the states' distribution depends on the θ owned by the *other* Gibbs component, so
# the CSMC reference has to stay pinned to its retained trajectory as the model is re-conditioned
# between sweeps. The exact θ posterior comes from quadrature against the closed-form likelihood, and
# the θ-mixed state marginals from the laws of total expectation and variance over the same grid.

"Draws for one variable as a plain vector; chain indexing yields an iteration×chain matrix."
particle_draws(chn, vn) = vec(collect(chn[vn]))

"Batch-means standard error of the mean of a correlated chain."
function batch_means_se(v; nbatches::Int=40)
    n = length(v) ÷ nbatches
    b = [mean(@view v[((i - 1) * n + 1):(i * n)]) for i in 1:nbatches]
    return std(b) / sqrt(nbatches)
end

"""
Assert that `samples` estimates `exact` to within `nsigma` batch-means standard errors. Using the
chain's own error estimate keeps the tolerance honest as mixing changes, rather than hard-coding an
`atol` that silently becomes either vacuous or flaky.
"""
function test_within_mc_error(exact, samples; nsigma=4)
    @test abs(mean(samples) - exact) <= nsigma * batch_means_se(samples)
    return nothing
end

# One model per SSM, with the parameter always sampled; the fixed-parameter tests `fix` it instead of
# duplicating the body. `fix` substitutes the value without adding a log-density term, so the sweep
# still sees one filtering step per observation -- `condition` would turn the assume into an observe
# and add a produce, which measurably changes the draws.
@model function lgssm(y, a, r)
    q ~ InverseGamma(3, 2)
    # `typeof(q)` stays generic when HMC differentiates through `q`, without boxing every element the
    # way `Vector{Real}` would.
    x = Vector{typeof(q)}(undef, length(y))
    x[1] ~ Normal(0, sqrt(q / (1 - a^2)))
    y[1] ~ Normal(x[1], sqrt(r))
    for t in 2:length(y)
        x[t] ~ Normal(a * x[t - 1], sqrt(q))
        y[t] ~ Normal(x[t], sqrt(r))
    end
end

const HMM_P = [0.80 0.15 0.05; 0.10 0.80 0.10; 0.05 0.15 0.80]
const HMM_MEANS = [-1.5, 0.0, 1.5]
const HMM_PI0 = ExactSSM.stationary_distribution(HMM_P)

@model function hmm(y)
    sd ~ LogNormal(log(0.7), 0.4)
    z = Vector{Int}(undef, length(y))
    z[1] ~ Categorical(HMM_PI0)
    y[1] ~ Normal(HMM_MEANS[z[1]], sd)
    for t in 2:length(y)
        z[t] ~ Categorical(HMM_P[z[t - 1], :])
        y[t] ~ Normal(HMM_MEANS[z[t]], sd)
    end
end

@testset "linear Gaussian SSM" begin
    ExactSSM.test_exact_ssm_reference()

    a, r, true_q, T = 0.8, 0.3, 0.5, 6
    s0(q) = q / (1 - a^2)                       # stationary, so the chain has no burn-in transient

    # Named `xtrue`, not `x`: a `@model` sharing a local scope with an `x` assignment captures it,
    # so every particle would mutate one shared array -- silently, and catastrophically.
    rng = StableRNG(1234)
    xtrue = zeros(T)
    xtrue[1] = sqrt(s0(true_q)) * randn(rng)
    for t in 2:T
        xtrue[t] = a * xtrue[t - 1] + sqrt(true_q) * randn(rng)
    end
    y = xtrue .+ sqrt(r) .* randn(rng, T)

    @testset "PG recovers the exact smoothing marginals" begin
        means, vars = ExactSSM.lgssm_smoother(y, a, true_q, r, s0(true_q))
        chn = sample(
            StableRNG(24), fix(lgssm(y, a, r), @varname(q) => true_q), PG(32), 4_000
        )
        for t in 1:T
            xs = particle_draws(chn, @varname(x[t]))
            test_within_mc_error(means[t], xs)
            test_within_mc_error(vars[t], (xs .- mean(xs)) .^ 2)
        end
    end

    @testset "Gibbs(q => NUTS, x => CSMC) recovers the exact posterior" begin
        prior = InverseGamma(3, 2)
        qs = range(0.05, 4.0; length=400)
        w = ExactSSM.grid_posterior(
            prior, qs, q -> ExactSSM.lgssm_loglik(y, a, q, r, s0(q))
        )
        q_mean, q_sd = ExactSSM.grid_moments(w, qs)
        smoothed = [ExactSSM.lgssm_smoother(y, a, q, r, s0(q)) for q in qs]
        x_mean = sum(w[i] * first(smoothed[i]) for i in eachindex(w))
        x_second = sum(
            w[i] * (last(smoothed[i]) .+ first(smoothed[i]) .^ 2) for i in eachindex(w)
        )

        alg = Gibbs(@varname(q) => NUTS(), @varname(x) => CSMC(32))
        chn = sample(StableRNG(31), lgssm(y, a, r), alg, 4_000)

        qd = particle_draws(chn, @varname(q))
        test_within_mc_error(q_mean, qd)
        @test std(qd) ≈ q_sd rtol = 0.2
        for t in 1:T
            xs = particle_draws(chn, @varname(x[t]))
            test_within_mc_error(x_mean[t], xs)
            @test var(xs) ≈ x_second[t] - x_mean[t]^2 rtol = 0.25
        end
    end
end

@testset "discrete HMM" begin
    P, π0, means = HMM_P, HMM_PI0, HMM_MEANS
    true_sd, K, T = 0.7, 3, 6
    obs_loglik(y, sd) = [logpdf(Normal(means[k], sd), y[t]) for t in 1:length(y), k in 1:K]

    # `ztrue`, not `z`, for the same reason as `xtrue` above.
    rng = StableRNG(99)
    ztrue = Vector{Int}(undef, T)
    ztrue[1] = rand(rng, Categorical(π0))
    for t in 2:T
        ztrue[t] = rand(rng, Categorical(P[ztrue[t - 1], :]))
    end
    y = [means[ztrue[t]] + true_sd * randn(rng) for t in 1:T]

    @testset "PG recovers the exact state marginals" begin
        # Discrete states make this sharp: the reference is a probability vector, so any bias shows
        # up directly instead of being absorbed into a mean.
        post, _ = ExactSSM.hmm_forward_backward(π0, P, obs_loglik(y, true_sd))
        chn = sample(StableRNG(25), fix(hmm(y), @varname(sd) => true_sd), PG(32), 4_000)
        for t in 1:T
            zs = particle_draws(chn, @varname(z[t]))
            for k in 1:K
                post[t, k] < 0.02 && continue  # too rare to resolve at this chain length
                test_within_mc_error(post[t, k], Float64.(zs .== k))
            end
        end
    end

    @testset "Gibbs(sd => HMC, z => CSMC) recovers the exact posterior" begin
        prior = LogNormal(log(0.7), 0.4)
        sds = range(0.2, 2.5; length=400)
        w = ExactSSM.grid_posterior(
            prior, sds, s -> last(ExactSSM.hmm_forward_backward(π0, P, obs_loglik(y, s)))
        )
        sd_mean, sd_sd = ExactSSM.grid_moments(w, sds)
        posts = [first(ExactSSM.hmm_forward_backward(π0, P, obs_loglik(y, s))) for s in sds]
        mixed = sum(w[i] * posts[i] for i in eachindex(w))

        alg = Gibbs(@varname(sd) => HMC(0.1, 12), @varname(z) => CSMC(32))
        chn = sample(StableRNG(32), hmm(y), alg, 4_000)

        sdd = particle_draws(chn, @varname(sd))
        test_within_mc_error(sd_mean, sdd)
        @test std(sdd) ≈ sd_sd rtol = 0.2
        for t in 1:T
            zs = particle_draws(chn, @varname(z[t]))
            for k in 1:K
                mixed[t, k] < 0.02 && continue
                test_within_mc_error(mixed[t, k], Float64.(zs .== k))
            end
        end
    end
end

end
