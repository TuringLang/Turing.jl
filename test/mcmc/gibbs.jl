module GibbsTests

using ..Models: MoGtest_default, MoGtest_default_z_vector, gdemo, gdemo_default
using ..NumericalTests:
    check_MoGtest_default,
    check_MoGtest_default_z_vector,
    check_gdemo,
    check_numerical,
    two_sample_test
import Combinatorics
using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
using Distributions: Distributions, InverseGamma, Normal
using Distributions: sample
using DynamicPPL: DynamicPPL
using FlexiChains: FlexiChains
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra
using Random: Random, Xoshiro
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_broken, @test_throws, @testset
using Turing
using Turing: Inference
using Turing.Inference: AdvancedHMC, AdvancedMH

const TuringDistributionsExt = Base.get_extension(Turing, :TuringDistributionsExt)
using .TuringDistributionsExt: ChineseRestaurantProcess, DirichletProcess

# Used by the models in two testsets below, which each used to define their own and so
# overwrote the type on load.
struct Wrapper{T<:Real}
    a::T
end

function check_transition_varnames(transition::DynamicPPL.ParamsWithStats, parent_varnames)
    for vn in keys(transition.params)
        @test any(Base.Fix2(DynamicPPL.subsumes, vn), parent_varnames)
    end
end

# A component differing from the sampler it wraps in exactly one way: it keeps the values of
# variables it is not currently sampling and reports them again, which a component may
# legitimately want to do to reuse them if the branch comes back. Gibbs must take the set of
# variables from the model rather than from this report, or such a component would keep a stale
# leaf in the snapshot and hide a change of support from `check_variable_set`.
struct KeepsValues{S} <: AbstractMCMC.AbstractSampler
    inner::S
end
struct KeepsValuesState{S,V}
    inner::S
    report::V
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, spl::KeepsValues; kwargs...
)
    transition, state = AbstractMCMC.step(rng, model, spl.inner; kwargs...)
    return transition, KeepsValuesState(state, Inference.gibbs_get_parameter_values(state))
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::KeepsValues,
    state::KeepsValuesState;
    kwargs...,
)
    transition, inner = AbstractMCMC.step(rng, model, spl.inner, state.inner; kwargs...)
    report = merge(state.report, Inference.gibbs_get_parameter_values(inner))
    return transition, KeepsValuesState(inner, report)
end
function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, spl::KeepsValues, state...; kwargs...
)
    return AbstractMCMC.step(rng, model, spl, state...; kwargs...)
end
Inference.gibbs_get_parameter_values(state::KeepsValuesState) = state.report
function Inference.gibbs_update_state!!(
    spl::KeepsValues, state::KeepsValuesState, model, gv
)
    return KeepsValuesState(
        Inference.gibbs_update_state!!(spl.inner, state.inner, model, gv), state.report
    )
end
Inference.gibbs_get_stats(state::KeepsValuesState) = Inference.gibbs_get_stats(state.inner)
function Inference.allow_varying_dimension(spl::KeepsValues)
    return Inference.allow_varying_dimension(spl.inner)
end
function Inference.allow_discrete_variables(spl::KeepsValues)
    return Inference.allow_discrete_variables(spl.inner)
end
Inference.init_strategy(spl::KeepsValues) = Inference.init_strategy(spl.inner)

@testset verbose = true "Gibbs conditioning" begin
    @testset "type stability" begin
        # A test model that has multiple features in one package:
        # Floats, Ints, arguments, observations, loops, dot_tildes.
        @model function test_model(obs1, obs2, num_vars, mean)
            variance ~ Exponential(2)
            z = Vector{Float64}(undef, num_vars)
            z .~ truncated(Normal(mean, variance); lower=1)
            y = Vector{Int64}(undef, num_vars)
            for i in 1:num_vars
                y[i] ~ Poisson(Int(round(z[i])))
            end
            s = sum(y) - sum(z)
            q = Wrapper(0.0)
            q.a ~ Normal(s, 1)
            r = Vector{Float64}(undef, 1)
            r[1] ~ Normal(q.a, 1)
            obs1 ~ Normal(r[1], 1)
            obs2 ~ Poisson(y[3])
            return obs1, obs2, variance, z, y, s
        end

        model = test_model(1.2, 2, 10, 2.5)
        all_varnames = DynamicPPL.VarName[
            @varname(variance), @varname(z), @varname(y), @varname(q.a), @varname(r[1])
        ]
        # All combinations of elements in all_varnames.
        target_vn_combinations = Iterators.flatten(
            Iterators.map(
                n -> Combinatorics.combinations(all_varnames, n), 1:length(all_varnames)
            ),
        )

        @testset "$(target_vns)" for target_vns in target_vn_combinations
            global_vnt = rand(model)
            target_vns = collect(target_vns)
            conditioned = Turing.Inference.conditioned_values(global_vnt, target_vns)

            # The component conditions on exactly the variables it does not sample.
            for k in keys(global_vnt)
                is_target = any(Iterators.map(vn -> DynamicPPL.subsumes(vn, k), target_vns))
                @test DynamicPPL.haskey(conditioned, k) == !is_target
            end

            # Check that init!! is type stable.
            conditioned_model = DynamicPPL.condition(model, conditioned)
            accs = DynamicPPL.OnlyAccsVarInfo()
            _, accs = @inferred DynamicPPL.init!!(
                conditioned_model, accs, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
            )
        end
    end
end

@testset "Invalid Gibbs constructor" begin
    # More samplers than varnames or vice versa
    @test_throws ArgumentError Gibbs((@varname(s), @varname(m)), (NUTS(), NUTS(), NUTS()))
    @test_throws ArgumentError Gibbs(
        (@varname(s), @varname(m), @varname(x)), (NUTS(), NUTS())
    )
    # Invalid samplers
    @test_throws ArgumentError Gibbs(@varname(s) => Emcee(10, 2.0))
    @test_throws ArgumentError Gibbs(
        @varname(s) => SGHMC(; learning_rate=0.01, momentum_decay=0.1)
    )
    @test_throws ArgumentError Gibbs(
        @varname(s) => SGLD(; stepsize=PolynomialStepsize(0.25))
    )
    # Values that we don't know how to convert to VarNames.
    @test_throws MethodError Gibbs(1 => NUTS())
    @test_throws MethodError Gibbs("x" => NUTS())
end

@testset "the deprecated `isgibbscomponent` is still honoured" begin
    struct OldStyleSampler <: AbstractMCMC.AbstractSampler end
    Turing.Inference.isgibbscomponent(::OldStyleSampler) = false
    @test !Turing.Inference.supports_gibbs(OldStyleSampler())
    @test_throws ArgumentError Gibbs(@varname(s) => OldStyleSampler())

    struct NewStyleSampler <: AbstractMCMC.AbstractSampler end
    @test Turing.Inference.supports_gibbs(NewStyleSampler())

    # A wrapper written against the old name delegates through it, so the old name has to
    # keep answering for Turing's own samplers rather than being a constant `true`.
    struct OldStyleWrapper{S} <: AbstractMCMC.AbstractSampler
        inner::S
    end
    Turing.Inference.isgibbscomponent(w::OldStyleWrapper) =
        Turing.Inference.isgibbscomponent(w.inner)
    @test !Turing.Inference.supports_gibbs(OldStyleWrapper(Prior()))
    @test Turing.Inference.supports_gibbs(OldStyleWrapper(MH()))
end

@testset "the deprecated `gibbs_get_raw_values` is still honoured" begin
    struct OldStyleState end
    Turing.Inference.gibbs_get_raw_values(::OldStyleState) =
        DynamicPPL.VarNamedTuple((; a=1.0))
    struct NewStyleState end
    Turing.Inference.gibbs_get_parameter_values(::NewStyleState) =
        DynamicPPL.VarNamedTuple((; b=2.0))
    struct NeitherState end

    # Gibbs asks by the new name, and a state written against the old one still answers.
    @test DynamicPPL.getvalue(
        Turing.Inference.gibbs_get_parameter_values(OldStyleState()), @varname(a)
    ) == 1.0
    # The old name also reaches a state that defines only the new one.
    @test DynamicPPL.getvalue(
        Turing.Inference.gibbs_get_raw_values(NewStyleState()), @varname(b)
    ) == 2.0
    # With neither defined, either name is a `MethodError` rather than a recursion.
    @test_throws MethodError Turing.Inference.gibbs_get_parameter_values(NeitherState())
    @test_throws MethodError Turing.Inference.gibbs_get_raw_values(NeitherState())
end

@testset "an unsupported component is named through its wrapper" begin
    for spl in (externalsampler(Prior()), Turing.Inference.RepeatSampler(SMC(), 2))
        err = try
            Gibbs(@varname(s) => spl)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        # The wrapper forwards `supports_gibbs`, so the sampler that answered is the inner one.
        @test occursin("Prior is not", err.msg) || occursin("SMC is not", err.msg)
    end
end

@testset "a component that keeps no linked layout needs no link" begin
    # `MH` rebuilds from the conditioned values each step, so `Gibbs` compares its block at the
    # values' own shape and never derives a linking transform for it. The distribution has to
    # move every sweep, since an unchanged one is settled without measuring either way.
    struct Unlinkable{T} <: Distributions.ContinuousUnivariateDistribution
        lo::T
    end
    Distributions.logpdf(d::Unlinkable, x) = Distributions.logpdf(Normal(d.lo, 1), x)
    function Distributions.rand(rng::Random.AbstractRNG, d::Unlinkable)
        return rand(rng, Normal(d.lo, 1))
    end
    Bijectors.VectorBijectors.to_linked_vec(::Unlinkable) = error("cannot be linked")

    @model function moving()
        a ~ Normal()
        return x ~ Unlinkable(a)
    end
    @test sample(
        Xoshiro(1), moving(), Gibbs(:a => MH(), :x => MH()), 20; progress=false
    ) isa Any
end

@testset "latent declared as a missing model argument" begin
    # Gibbs conditions every non-target variable, and conditioning cannot override an argument
    # bound to `missing`: the `missing` reaches the likelihood. Refused with the argument named,
    # rather than left to throw `MethodError` from inside `loglikelihood`.
    @model function impute(x)
        m ~ Normal(0, 1)
        x ~ Normal(m, 1)
        return 2.0 ~ Normal(x, 1)
    end
    @test_throws "is or holds `missing`" sample(
        StableRNG(468),
        impute(missing),
        Gibbs(@varname(m) => MH(), @varname(x) => MH()),
        10;
        progress=false,
    )
    # An element of an array argument goes the same way as a whole one, and a keyword argument
    # lands in `model.defaults` rather than `model.args`. Both reached the likelihood.
    @model function partly(x)
        mu ~ Normal()
        for i in eachindex(x)
            x[i] ~ Normal(mu, 1)
        end
    end
    @test_throws "is or holds `missing`" sample(
        StableRNG(468),
        partly([1.5, missing]),
        Gibbs(@varname(mu) => MH(), @varname(x[2]) => MH()),
        10;
        check_model=false,
        progress=false,
    )
    # A nested container hides a `missing` from a check that reads `eltype` alone, since the
    # outer element type is the inner array, never `>: Missing`.
    @model function nested(x)
        mu ~ Normal()
        for v in x, e in v
            e ~ Normal(mu, 1)
        end
    end
    @test_throws "is or holds `missing`" sample(
        StableRNG(468),
        nested([[1.5, missing]]),
        Gibbs(@varname(mu) => MH()),
        10;
        check_model=false,
        progress=false,
    )
    @model function bykeyword(a; b=missing)
        m ~ Normal()
        a ~ Normal(m, 1)
        return b ~ Normal(m, 1)
    end
    @test_throws "is or holds `missing`" sample(
        StableRNG(468),
        bykeyword(1.0),
        Gibbs(@varname(m) => MH(), @varname(b) => MH()),
        10;
        check_model=false,
        progress=false,
    )
    # Gibbs's restriction, not the model's: another sampler takes it, and so does Gibbs once
    # the variable is declared in the model rather than taken as an argument.
    @test mean(
        sample(StableRNG(468), impute(missing), MH(), 2000; progress=false)[@varname(m)]
    ) ≈ 2 / 3 atol = 0.15
    @model function impute_inner()
        m ~ Normal(0, 1)
        x ~ Normal(m, 1)
        return 2.0 ~ Normal(x, 1)
    end
    chn = sample(
        StableRNG(468),
        impute_inner(),
        Gibbs(@varname(m) => MH(), @varname(x) => MH()),
        2000;
        progress=false,
    )
    # Conjugate posterior: E[m] = 2/3, E[x] = 4/3.
    @test mean(chn[@varname(m)]) ≈ 2 / 3 atol = 0.15
    @test mean(chn[@varname(x)]) ≈ 4 / 3 atol = 0.15
end

@testset "Missing Gibbs samplers throw error" begin
    @model function gdemo_missing(x, y)
        s² ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s²))
        x ~ Normal(m, sqrt(s²))
        return y ~ Normal(m, sqrt(s²))
    end
    model = gdemo_missing(1.5, 2.0)

    # If a variable has no component sampler it is never updated.
    @test_throws ArgumentError sample(model, Gibbs(:m => MH()), 10)

    # `check_model=false` turns off the model diagnostics, not this: a variable no component
    # samples is conditioned on one draw for the whole run, never what was meant.
    @test_throws ArgumentError sample(
        model, Gibbs(:m => MH()), 10; check_model=false, progress=false
    )
end

# Test that the samplers are being called in the correct order, on the correct target
# variables.
#
@testset "Sampler call order" begin
    # A wrapper around inference algorithms to allow intercepting the dispatch cascade to
    # collect testing information.
    struct AlgWrapper{Alg<:AbstractMCMC.AbstractSampler} <: AbstractMCMC.AbstractSampler
        inner::Alg
    end

    # Methods we need to define to be able to use AlgWrapper instead of an actual algorithm.
    # They all just propagate the call to the inner algorithm.
    Turing.Inference.supports_gibbs(wrap::AlgWrapper) =
        Turing.Inference.supports_gibbs(wrap.inner)
    function Turing.Inference.gibbs_update_state!!(
        sampler::AlgWrapper,
        state,
        model::DynamicPPL.Model,
        global_vnt::DynamicPPL.VarNamedTuple,
    )
        return Turing.Inference.gibbs_update_state!!(
            sampler.inner, state, model, global_vnt
        )
    end

    # conditioned_and_algs records, for every component step, the variables that component
    # was conditioned on and the sampler that ran. It is filled by the `step` method below.
    conditioned_and_algs = Any[]

    # The methods that capture testing information for us.
    function AbstractMCMC.step(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        sampler::AlgWrapper,
        args...;
        kwargs...,
    )
        push!(
            conditioned_and_algs,
            (keys(DynamicPPL.conditioned(model.context)), sampler.inner),
        )
        return AbstractMCMC.step(rng, model, sampler.inner, args...; kwargs...)
    end

    # A test model that includes several different kinds of tilde syntax.
    @model function test_model(val, (::Type{M})=Vector{Float64}) where {M}
        s ~ Normal(0.1, 0.2)
        m ~ Poisson()
        val ~ Normal(s, 1)
        1.0 ~ Normal(s + m, 1)

        n := m
        xs = M(undef, 5)
        for i in eachindex(xs)
            xs[i] ~ Beta(0.5, 0.5)
        end

        ys = M(undef, 2)
        ys .~ Beta(1.0, 1.0)

        q = Wrapper(0.0)
        q.a ~ Normal(s, 1)
        r = M(undef, 1)
        r[1] ~ Normal(q.a, 1)

        return sum(xs), sum(ys), n
    end

    mh = MH()
    pg = PG(10)
    hmc = HMC(0.01, 4)
    nuts = NUTS()
    # Sample with all sorts of combinations of samplers and targets.
    sampler = Gibbs(
        @varname(s) => AlgWrapper(mh),
        (@varname(s), @varname(m)) => AlgWrapper(mh),
        @varname(m) => AlgWrapper(pg),
        @varname(xs) => AlgWrapper(hmc),
        @varname(ys) => AlgWrapper(nuts),
        @varname(q) => AlgWrapper(hmc),
        @varname(r) => AlgWrapper(hmc),
        @varname(ys) => AlgWrapper(nuts),
        (@varname(xs), @varname(ys)) => AlgWrapper(hmc),
        @varname(s) => AlgWrapper(mh),
        @varname(q.a) => AlgWrapper(mh),
        @varname(r[1]) => AlgWrapper(mh),
    )
    chain = sample(test_model(-1), sampler, 2)

    expected_targets_and_algs_per_iteration = [
        ((@varname(s),), mh),
        ((@varname(s), @varname(m)), mh),
        ((@varname(m),), pg),
        ((@varname(xs),), hmc),
        ((@varname(ys),), nuts),
        ((@varname(q),), hmc),
        ((@varname(r),), hmc),
        ((@varname(ys),), nuts),
        ((@varname(xs), @varname(ys)), hmc),
        ((@varname(s),), mh),
        ((@varname(q.a),), mh),
        ((@varname(r[1]),), mh),
    ]
    expected = vcat(
        expected_targets_and_algs_per_iteration, expected_targets_and_algs_per_iteration
    )
    @test length(conditioned_and_algs) == length(expected)
    # Every random variable of the model, so that conditioning can be checked for
    # completeness and not just for soundness.
    model_vns = [
        @varname(s), @varname(m), @varname(xs), @varname(ys), @varname(q.a), @varname(r[1])
    ]
    # Each component runs the expected sampler, and is conditioned on every variable except
    # the ones it samples.
    for ((targets, alg), (conditioned, actual_alg)) in zip(expected, conditioned_and_algs)
        @test actual_alg === alg
        overlaps(a, b) = DynamicPPL.subsumes(a, b) || DynamicPPL.subsumes(b, a)
        # Sound: nothing the component samples is conditioned, or it could not move.
        @test !any(t -> any(k -> overlaps(t, k), conditioned), targets)
        # Complete: everything it does not sample is conditioned, or it would be left free
        # to resample a variable another component owns.
        for vn in model_vns
            any(t -> overlaps(vn, t), targets) && continue
            @test any(k -> overlaps(vn, k), conditioned)
        end
    end
end

@testset "Equivalence of RepeatSampler and repeating Sampler" begin
    sampler1 = Gibbs(@varname(s) => RepeatSampler(MH(), 3), @varname(m) => ESS())
    sampler2 = Gibbs(
        @varname(s) => MH(), @varname(s) => MH(), @varname(s) => MH(), @varname(m) => ESS()
    )
    chain1 = sample(Xoshiro(23), gdemo_default, sampler1, 10)
    chain2 = sample(Xoshiro(23), gdemo_default, sampler1, 10)
    @test FlexiChains.has_same_data(chain1, chain2)
end

@testset "Gibbs warmup" begin
    # An inference algorithm, for testing purposes, that records how many warm-up steps
    # and how many non-warm-up steps haven been taken.
    mutable struct WarmupCounter <: AbstractMCMC.AbstractSampler
        warmup_init_count::Int
        non_warmup_init_count::Int
        warmup_count::Int
        non_warmup_count::Int

        WarmupCounter() = new(0, 0, 0, 0)
    end

    Turing.Inference.supports_gibbs(::WarmupCounter) = true

    # A component has to report the variables it samples; reporting nothing would tell Gibbs
    # the model stopped reaching them. This counter only measures how often it is called, so
    # one draw is made and then reused rather than re-evaluating the model every step.
    function _counter_values(rng, model, state)
        state === nothing || return state.values
        accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
        _, accs = DynamicPPL.init!!(
            rng, model, accs, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
        )
        return DynamicPPL.get_parameter_values(accs)
    end

    # we need some state type to implement the Gibbs interface (we can't just use `nothing`)
    struct TrivialState{V<:DynamicPPL.VarNamedTuple}
        values::V
    end
    Turing.Inference.gibbs_get_parameter_values(s::TrivialState) = s.values
    function Turing.Inference.gibbs_update_state!!(
        ::WarmupCounter, s::TrivialState, ::DynamicPPL.Model, ::DynamicPPL.VarNamedTuple
    )
        return s
    end

    function AbstractMCMC.step(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        spl::WarmupCounter,
        state::Union{Nothing,TrivialState}=nothing;
        kwargs...,
    )
        if state === nothing
            spl.non_warmup_init_count += 1
        else
            spl.non_warmup_count += 1
        end
        # no need a transition since we never check the actual outputs
        return nothing, TrivialState(_counter_values(rng, model, state))
    end

    function AbstractMCMC.step_warmup(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        spl::WarmupCounter,
        state::Union{Nothing,TrivialState}=nothing;
        kwargs...,
    )
        if state === nothing
            spl.warmup_init_count += 1
        else
            spl.warmup_count += 1
        end
        return nothing, TrivialState(_counter_values(rng, model, state))
    end

    @model f() = x ~ Normal()
    m = f()

    num_samples = 10
    num_warmup = 3
    wuc = WarmupCounter()
    sample(m, Gibbs(:x => wuc), num_samples; num_warmup=num_warmup)
    @test wuc.warmup_init_count == 1
    @test wuc.non_warmup_init_count == 0
    @test wuc.warmup_count == num_warmup
    @test wuc.non_warmup_count == num_samples - 1

    num_reps = 2
    wuc = WarmupCounter()
    sample(m, Gibbs(:x => RepeatSampler(wuc, num_reps)), num_samples; num_warmup=num_warmup)
    @test wuc.warmup_init_count == 1
    @test wuc.non_warmup_init_count == 0
    @test wuc.warmup_count == num_warmup * num_reps
    @test wuc.non_warmup_count == (num_samples - 1) * num_reps
end

@testset verbose = true "Testing gibbs.jl" begin
    @info "Starting Gibbs tests"

    @testset "Gibbs constructors" begin
        # Create Gibbs samplers with various configurations and ways of passing the
        # arguments, and run them all on the `gdemo_default` model, see that nothing breaks.
        N = 10
        # Two variables being sampled by one sampler.
        s1 = Gibbs((@varname(s), @varname(m)) => HMC(0.1, 5))
        s2 = Gibbs((@varname(s), :m) => PG(10))
        # As above but different samplers and using kwargs.
        s3 = Gibbs(:s => CSMC(3), :m => HMCDA(200, 0.65, 0.15))
        s4 = Gibbs(@varname(s) => HMC(0.1, 5), @varname(m) => ESS())
        # Multiple instances of the same sampler. This implements running, in this case,
        # 3 steps of HMC on m and 2 steps of PG on m in every iteration of Gibbs.
        s5 = begin
            hmc = HMC(0.1, 5)
            pg = PG(10)
            vns = @varname(s)
            vnm = @varname(m)
            Gibbs(vns => hmc, vns => hmc, vns => hmc, vnm => pg, vnm => pg)
        end
        # Same thing but using RepeatSampler.
        s6 = Gibbs(
            @varname(s) => RepeatSampler(HMC(0.1, 5), 3),
            @varname(m) => RepeatSampler(PG(10), 2),
        )
        @test sample(gdemo_default, s1, N) isa VNChain
        @test sample(gdemo_default, s2, N) isa VNChain
        @test sample(gdemo_default, s3, N) isa VNChain
        @test sample(gdemo_default, s4, N) isa VNChain
        @test sample(gdemo_default, s5, N) isa VNChain
        @test sample(gdemo_default, s6, N) isa VNChain
    end

    # Test various combinations of samplers against models for which we know the analytical
    # posterior mean.
    @testset "Gibbs inference" begin
        @testset "CSMC and HMC on gdemo" begin
            alg = Gibbs(:s => CSMC(15), :m => HMC(0.2, 4))
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_numerical(chain, [@varname(m)], [7 / 6]; atol=0.15)
            # Be more relaxed with the tolerance of the variance.
            check_numerical(chain, [@varname(s)], [49 / 24]; atol=0.35)
        end

        @testset "MH and HMCDA on gdemo" begin
            alg = Gibbs(:s => MH(), :m => HMCDA(200, 0.65, 0.3))
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_gdemo(chain; atol=0.1)
        end

        @testset "CSMC and ESS on gdemo" begin
            alg = Gibbs(:s => CSMC(15), :m => ESS())
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_gdemo(chain; atol=0.1)
        end

        # TODO(mhauru) Why is this in the Gibbs test suite?
        @testset "CSMC on gdemo" begin
            alg = CSMC(15)
            chain = sample(gdemo(1.5, 2.0), alg, 4_000)
            check_gdemo(chain; atol=0.1)
        end

        @testset "PG and HMC on MoGtest_default" begin
            gibbs = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => PG(15),
                (@varname(mu1), @varname(mu2)) => HMC(0.15, 3),
            )
            chain = sample(MoGtest_default, gibbs, 2_000)
            check_MoGtest_default(chain; atol=0.15)
        end

        @testset "Multiple overlapping samplers on gdemo" begin
            # Test samplers that are run multiple times, or have overlapping targets.
            alg = Gibbs(
                @varname(s) => MH(),
                (@varname(s), @varname(m)) => MH(),
                @varname(m) => ESS(),
                @varname(s) => RepeatSampler(MH(), 3),
                @varname(m) => HMC(0.2, 4),
                (@varname(m), @varname(s)) => HMC(0.2, 4),
            )
            chain = sample(gdemo(1.5, 2.0), alg, 500)
            check_gdemo(chain; atol=0.15)
        end

        @testset "Multiple overlapping samplers on MoGtest_default" begin
            gibbs = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => PG(15),
                (@varname(z1), @varname(z2)) => PG(15),
                (@varname(mu1), @varname(mu2)) => HMC(0.15, 3),
                (@varname(z3), @varname(z4)) => RepeatSampler(PG(15), 2),
                (@varname(mu1)) => ESS(),
                (@varname(mu2)) => ESS(),
                (@varname(z1), @varname(z2)) => PG(15),
            )
            chain = sample(MoGtest_default, gibbs, 500)
            check_MoGtest_default(chain; atol=0.15)
        end
    end

    @testset "transitions" begin
        @model function gdemo_copy()
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            1.5 ~ Normal(m, sqrt(s))
            2.0 ~ Normal(m, sqrt(s))
            return s, m
        end
        model = gdemo_copy()

        @nospecialize function AbstractMCMC.bundle_samples(
            samples::Vector, ::typeof(model), ::Gibbs, state, ::Type{VNChain}; kwargs...
        )
            samples isa Vector{<:DynamicPPL.ParamsWithStats} ||
                error("incorrect transitions")
            return nothing
        end

        function callback(rng, model, sampler, sample, state, i; kwargs...)
            sample isa DynamicPPL.ParamsWithStats || error("incorrect sample")
            return nothing
        end

        alg = Gibbs(:s => MH(), :m => HMC(0.2, 4))
        sample(model, alg, 100; callback=callback)
    end

    @testset "dynamic model needs a sampler built for varying dimension" begin
        # `b` decides whether `θ[2]` exists, so the dimension of the target changes between
        # sweeps. Only a sampler that redraws whatever the model reaches can sample that block.
        @model function dynamic_bernoulli_normal(y_obs=2.0)
            b ~ Bernoulli(0.3)

            θ = zeros(2)
            if b == 0
                θ[1] ~ Normal(0.0, 1.0)
                y_obs ~ Normal(θ[1], 0.5)
            else
                θ[1] ~ Normal(0.0, 1.0)
                θ[2] ~ Normal(0.0, 1.0)
                y_obs ~ Normal(θ[1] + θ[2], 0.5)
            end
        end
        model = dynamic_bernoulli_normal(2.0)

        # Splitting `b` off is rejected whatever samples `θ`, and for the same reason each
        # time: `b`'s step is what makes `θ[2]` come and go, while the `θ` component is the one
        # that samples it. The ownership condition decides this before the component's own
        # capability is consulted, so the message is the same for `MH`, `HMC` and `PG`, and
        # asserting it is what distinguishes the guard from an internal error raised further
        # down. Sampling this partition gave P(b=1) = 0.11 against an exact 0.394.
        for component in (MH(), HMC(0.1, 5), PG(50))
            @test_throws "which does not sample it" sample(
                StableRNG(42),
                model,
                Gibbs(:b => MH(), :θ => component),
                200;
                progress=false,
            )
        end

        # `Xoshiro(1)`'s initialising draw already reaches `θ[2]`, so `b`'s step makes it
        # *stop* existing rather than appear. It is load-bearing that this seed starts with
        # `b == 1`; `StableRNG(42)` above starts with `b == 0` and exercises the other
        # direction. Both are the ownership condition, so both name the deciding component.
        @test_throws "stopped existing during a step of the component sampling b" sample(
            Xoshiro(1), model, Gibbs(:b => MH(), :θ => PG(50)), 200; progress=false
        )

        # A component that keeps and reports the values it is not currently sampling must not
        # escape either check. Gibbs reads the set of variables off the model, so the verdict
        # is the same as for the bare sampler and does not depend on the initialising draw.
        # Reporting the stale leaf instead once let this partition sample, returning P(b=1)
        # between 0.08 and 0.18, but only for the seeds whose first draw reached `θ[2]`.
        for rng in (StableRNG(42), Xoshiro(1))
            @test_throws "which does not sample it" sample(
                rng,
                model,
                Gibbs(:b => MH(), :θ => KeepsValues(PG(20))),
                200;
                progress=false,
            )
        end

        # Being handed a block at a new shape is a different question from
        # `allow_varying_dimension`: here `PG` moves between the two supports and `θ`'s
        # component only ever sees a block already at one shape or the other, so each kernel
        # is applied at one fixed shape and the scheme is valid. Whether a component may is
        # settled by whether it implements `gibbs_update_state!!` for a `ReshapedBlock`, which
        # turns on what it carries between steps rather than on the algorithm.
        pg_block = (@varname(b), @varname(θ)) => PG(20)
        # `NUTS(0, δ)` and `HMCDA(0, δ, λ)` carry `NoAdaptation`, so they have nothing sized
        # for the block and take a reshaped one through the same method as `HMC`. Declaring it
        # on `StaticHamiltonian` alone refused them a block they can sample.
        for component in (MH(), HMC(0.1, 5), NUTS(0, 0.65), HMCDA(0, 0.65, 1.0))
            chn = sample(
                StableRNG(42),
                model,
                Gibbs(pg_block, @varname(θ) => component),
                2000;
                discard_initial=1000,
                progress=false,
            )
            bs = vec(chn[@varname(b)])
            @test count(==(1), bs) / length(bs) ≈ 0.394 atol = 0.05
        end

        # `NUTS` and `HMCDA` implement no such method: `gen_metric` renews their metric from
        # the adaptor, whose mass matrix is sized for the shape it adapted to, and
        # `AdvancedHMC` then reports an `AxesMismatch`. Nor does `ESS`, its prior means being
        # gathered for the block it was built on. Both are refused before being asked to step.
        for component in (NUTS(), ESS())
            @test_throws "does not implement" sample(
                StableRNG(42),
                model,
                Gibbs(pg_block, @varname(θ) => component),
                200;
                progress=false,
            )
        end

        # `externalsampler` answers from the wrapper's own space, not the sampler inside: with
        # `unconstrained=false` the wrapper works in unlinked space, so its block must not be
        # compared in linked space it never uses.
        @test !Turing.Inference.keeps_linked_layout(
            externalsampler(
                AdvancedMH.RWMH(MvNormal(zeros(1), LinearAlgebra.I)); unconstrained=false
            ),
        )
        @test Turing.Inference.keeps_linked_layout(
            externalsampler(
                AdvancedMH.RWMH(MvNormal(zeros(1), LinearAlgebra.I)); unconstrained=true
            ),
        )

        # A block can change shape with its leaf set unchanged, when a component that shares
        # the variable moves it to one that links to a different width: `x` has three leaves
        # either way, but a simplex occupies two numbers and a free vector three. The gate
        # compares the linked width, so this reaches the capability check rather than dying
        # inside `AdvancedHMC` with a bare `DimensionMismatch`.
        @model function relinked(y)
            b ~ Bernoulli(0.5)
            if b == 1
                x ~ Dirichlet(3, 1.0)
            else
                x ~ MvNormal(zeros(3), LinearAlgebra.I)
            end
            return y ~ Normal(sum(x), 1.0)
        end
        shared_block = (@varname(b), @varname(x)) => MH()
        @test_throws "does not implement" sample(
            StableRNG(42),
            relinked(1.0),
            Gibbs(shared_block, @varname(x) => NUTS()),
            200;
            progress=false,
        )
        # A component that does implement it rebuilds and samples.
        @test sample(
            StableRNG(42),
            relinked(1.0),
            Gibbs(shared_block, @varname(x) => HMC(0.05, 5)),
            200;
            progress=false,
        ) isa Any
        # The tilde KEYS can change while the leaves do not: one branch writes `x` in a single
        # tilde, the other writes `x[i]` in three. The two layouts are then disjoint, and a gate
        # that only compared keys present on both sides saw nothing while the linked dimension
        # moved from two to three, so `NUTS` died inside `AdvancedHMC` instead of being refused.
        @model function keyswap(y)
            b ~ Bernoulli(0.5)
            if b
                x ~ Dirichlet(3, 1.0)
            else
                x = Vector{Float64}(undef, 3)
                for i in 1:3
                    x[i] ~ Normal(0, 1)
                end
            end
            return y ~ Normal(sum(x), 1.0)
        end
        keyswap_block = (@varname(b), @varname(x)) => PG(10)
        # `NUTS` only: `ESS` refuses this model before the gate is reached, for a reason of its
        # own -- a `Dirichlet` is not a Gaussian prior -- so it would not test what this does.
        @test_throws "does not implement" sample(
            StableRNG(8),
            keyswap(1.0),
            Gibbs(keyswap_block, @varname(x) => NUTS()),
            100;
            progress=false,
        )
        @test sample(
            StableRNG(8),
            keyswap(1.0),
            Gibbs(keyswap_block, @varname(x) => HMC(0.05, 5)),
            100;
            progress=false,
        ) isa Any

        # A bound that moves within one transform type is not a shape change, and must not be
        # gated: the linked dimension never moves, so an adapting `NUTS` copes.
        @model function movingbound()
            a ~ Uniform(-1.0, 0.0)
            x ~ truncated(Normal(0, 1); lower=a)
            return 0.3 ~ Normal(x, 0.5)
        end
        @test sample(
            StableRNG(42),
            movingbound(),
            Gibbs((@varname(a), @varname(x)) => MH(), @varname(x) => NUTS()),
            200;
            progress=false,
        ) isa Any

        # A block that empties, rather than merely shrinking, is refused for every sampler.
        # `MH` would in fact sample it correctly, but a `LogDensityFunction` over no variables
        # is ill-formed and fails deep inside, so this is refused uniformly and loudly until
        # skipping such a component is implemented properly. Both directions are covered:
        # `Xoshiro(4)`'s draw starts with the block empty, caught before the first step, and
        # `Xoshiro(1)`'s starts non-empty, caught in a later sweep.
        @model function can_empty()
            b ~ Bernoulli(0.5)
            if b == 1
                w ~ Normal()
                1.0 ~ Normal(w, 0.5)
            else
                1.0 ~ Normal(0.0, 0.5)
            end
        end
        for rng in (Xoshiro(4), Xoshiro(1)), component in (MH(), HMC(0.1, 5))
            @test_throws "has nothing to sample" sample(
                rng,
                can_empty(),
                Gibbs((@varname(b), @varname(w)) => PG(20), @varname(w) => component),
                300;
                progress=false,
            )
        end

        # The capability is the wrapped sampler's, so `RepeatSampler` delegates to it.
        chn = sample(
            StableRNG(42),
            model,
            Gibbs(pg_block, @varname(θ) => RepeatSampler(HMC(0.1, 5), 2)),
            2000;
            discard_initial=1000,
            progress=false,
        )
        bs = vec(chn[@varname(b)])
        @test count(==(1), bs) / length(bs) ≈ 0.394 atol = 0.05
        @test_throws "does not implement" sample(
            StableRNG(42),
            model,
            Gibbs(pg_block, @varname(θ) => RepeatSampler(NUTS(), 2)),
            200;
            progress=false,
        )

        # Rejected even when both components can handle a varying set of variables: `θ[2]`
        # comes and goes under the `b` component's proposal while the `θ` component is the one
        # that samples it, so that component conditions on a variable the proposed state does
        # not have. Being capable is not enough; the two have to share a block.
        err = try
            sample(StableRNG(42), model, Gibbs(:b => PG(20), :θ => PG(20)), 200; progress=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        # The ownership condition, not the capability one: both components declare they can
        # sample a varying set, so a message about that would mean the wrong guard fired.
        @test occursin("which does not sample it", err.msg)

        # `b` and `θ` in one block, sampled by `PG`, which rebuilds its trace each sweep.
        # P(b=1 | y=2) = 0.3 * N(2; 0, sqrt(2.25)) normalised against 0.7 * N(2; 0, sqrt(1.25)).
        chn = sample(
            StableRNG(42),
            model,
            Gibbs((@varname(b), @varname(θ)) => PG(50)),
            2000;
            discard_initial=1000,
            progress=false,
        )
        @test size(chn, 1) == 2000
        bs = vec(chn[@varname(b)])
        @test count(==(1), bs) / length(bs) ≈ 0.394 atol = 0.05

        # The posterior above is only evidence about disappearance and reappearance if the
        # chain actually did both, so check that it moved each way rather than sitting in one
        # dimension and happening to land near 0.394.
        transitions = collect(zip(bs[1:(end - 1)], bs[2:end]))
        @test (1, 0) in transitions
        @test (0, 1) in transitions

        # Both states of `b` are visited, so the chain is not stuck in one dimension.
        @test length(unique(skipmissing(chn[@varname(b)]))) == 2

        theta1_samples = collect(skipmissing(chn[@varname(θ[1])]))
        @test all(isfinite, theta1_samples)
        @test std(theta1_samples) > 0.1

        # `θ[2]` exists only while `b == 1`, so the chain holds both missing and present values.
        theta2_samples = chn[@varname(θ[2])]
        @test any(ismissing, theta2_samples)
        @test any(!ismissing, theta2_samples)
    end

    @testset "Demo model" begin
        @testset verbose = true "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            vns = (@varname(m), @varname(s))
            samplers = [
                Gibbs(@varname(s) => NUTS(), @varname(m) => NUTS()),
                Gibbs(@varname(s) => NUTS(), @varname(m) => HMC(0.01, 4)),
                Gibbs(@varname(s) => NUTS(), @varname(m) => ESS()),
                Gibbs(@varname(s) => HMC(0.01, 4), @varname(m) => MH()),
                Gibbs(@varname(s) => MH(), @varname(m) => HMC(0.01, 4)),
            ]

            @testset "$sampler" for sampler in samplers
                # Check that taking steps performs as expected.
                rng = Random.default_rng()
                transition, state = AbstractMCMC.step(rng, model, sampler)
                check_transition_varnames(transition, vns)
                for _ in 1:5
                    transition, state = AbstractMCMC.step(rng, model, sampler, state)
                    check_transition_varnames(transition, vns)
                end
            end

            # Run the Gibbs sampler and NUTS on the same model, compare statistics of the
            # chains.
            @testset "comparison with 'gold-standard' samples" begin
                num_iterations = 2_000
                thinning = 10
                num_chains = 4

                # Determine initial parameters to make comparison as fair as possible.
                # posterior_mean returns a NamedTuple so we can plug it in directly.
                posterior_mean = DynamicPPL.TestUtils.posterior_mean(model)
                initial_params = fill(InitFromParams(posterior_mean), num_chains)

                # Sampler to use for Gibbs components.
                hmc = HMC(0.1, 32)
                sampler = Gibbs(@varname(s) => hmc, @varname(m) => hmc)
                chain = sample(
                    StableRNG(42),
                    model,
                    sampler,
                    MCMCThreads(),
                    num_iterations,
                    num_chains;
                    progress=false,
                    initial_params=initial_params,
                    discard_initial=1_000,
                    thinning=thinning,
                )

                # "Ground truth" samples.
                # TODO: Replace with closed-form sampling once that is implemented in DynamicPPL.

                chain_true = sample(
                    StableRNG(42),
                    model,
                    NUTS(),
                    MCMCThreads(),
                    num_iterations,
                    num_chains;
                    progress=false,
                    initial_params=initial_params,
                    thinning=thinning,
                )

                # Extract varname leaves.
                vns = DynamicPPL.TestUtils.varnames(model)
                vn_leaves = Set{DynamicPPL.VarName}()
                for vn in vns
                    val = first(chain[vn])
                    leaves = AbstractPPL.varname_leaves(vn, val)
                    vn_leaves = union(vn_leaves, leaves)
                end

                # Perform KS test to ensure that the chains are similar.
                for vn in vn_leaves
                    vals = vec(chain[vn])
                    true_vals = vec(chain_true[vn])
                    @test two_sample_test(vals, true_vals; warn_on_fail=true)
                    # Let's make sure that the significance level is not too low by
                    # checking that the KS test fails for some simple transformations.
                    # TODO: Replace the heuristic below with closed-form implementations
                    # of the targets, once they are implemented in DynamicPPL.
                    @test !two_sample_test(0.9 .* true_vals, true_vals)
                    @test !two_sample_test(1.1 .* true_vals, true_vals)
                    @test !two_sample_test(1e-1 .+ true_vals, true_vals)
                end
            end
        end
    end

    @testset "multiple varnames" begin
        @testset "with both `s` and `m` as random" begin
            model = gdemo(1.5, 2.0)
            vns = (@varname(s), @varname(m))
            spl = Gibbs(vns => MH())

            # `step`
            rng = Random.default_rng()
            transition, state = AbstractMCMC.step(rng, model, spl)
            check_transition_varnames(transition, vns)
            for _ in 1:5
                transition, state = AbstractMCMC.step(rng, model, spl, state)
                check_transition_varnames(transition, vns)
            end

            # `sample`
            chain = sample(StableRNG(42), model, spl, 1_000; progress=false)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.4)
        end

        @testset "without `m` as random" begin
            model = gdemo(1.5, 2.0) | (m=7 / 6,)
            vns = (@varname(s),)
            spl = Gibbs(vns => MH())

            # `step`
            rng = Random.default_rng()
            transition, state = AbstractMCMC.step(rng, model, spl)
            check_transition_varnames(transition, vns)
            for _ in 1:5
                transition, state = AbstractMCMC.step(rng, model, spl, state)
                check_transition_varnames(transition, vns)
            end
        end
    end

    @testset "statistic names that would collide are refused" begin
        struct FakeStatState{S}
            stats::S
        end
        Turing.Inference.gibbs_get_stats(s::FakeStatState) = s.stats

        # Joining a component's variables and a statistic's name with `_` is not injective:
        # prefix `x` with statistic `y_z`, and prefix `x_y` with statistic `z`, both give
        # `x_y_z`, and the merge used to keep only the second.
        spl = Gibbs(@varname(x) => MH(), @varname(x_y) => MH())
        states = (FakeStatState((; y_z=1)), FakeStatState((; z=2)))
        err = try
            Turing.Inference.component_stats(spl, states)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("x_y_z", err.msg)

        # Names that do not collide are still merged.
        ok = Turing.Inference.component_stats(
            spl, (FakeStatState((; a=1)), FakeStatState((; b=2)))
        )
        @test ok == (x_a=1, x_y_b=2)
    end

    @testset "component sampler stats reach the chain" begin
        @model function two_normals()
            h ~ Normal()
            m ~ Normal()
            return 0.0 ~ Normal(h + m)
        end
        chn = sample(
            StableRNG(468),
            two_normals(),
            Gibbs(@varname(h) => HMC(0.1, 5), @varname(m) => MH()),
            10;
            progress=false,
        )
        # HMC's diagnostics are prefixed with the variables that component samples. As for a
        # standalone HMC chain, the initial step has no stats, so the first entry is missing.
        acceptance = collect(skipmissing(vec(chn[Symbol("h_acceptance_rate")])))
        @test length(acceptance) == 9
        @test all(0 .<= acceptance .<= 1)
        @test all(>(0), collect(skipmissing(vec(chn[Symbol("h_n_steps")]))))
        accepted = collect(skipmissing(vec(chn[Symbol("m_accepted")])))
        @test length(accepted) == 10
        @test all(a -> a isa Bool, accepted)

        # Components sampling the same variables are distinguished by their index.
        chn2 = sample(
            StableRNG(468),
            two_normals(),
            Gibbs(
                @varname(h) => HMC(0.1, 5), @varname(h) => HMC(0.1, 5), @varname(m) => MH()
            ),
            5;
            progress=false,
        )
        @test !isempty(collect(skipmissing(vec(chn2[Symbol("h_1_acceptance_rate")]))))
        @test !isempty(collect(skipmissing(vec(chn2[Symbol("h_2_acceptance_rate")]))))

        # Statistic names must not parse as variable names: chain packages read variable
        # structure back out of them, so a stat called `x[1]_acceptance_rate` would be served
        # up as one of `x`'s draws (`MCMCChains.namesingroup(chn, :x)` matches `x[`).
        @model function indexed()
            x = Vector{Float64}(undef, 2)
            x[1] ~ Normal()
            x[2] ~ Normal()
            return 0.0 ~ Normal(x[1] + x[2])
        end
        chn3 = sample(
            StableRNG(468),
            indexed(),
            Gibbs(@varname(x[1]) => HMC(0.1, 5), @varname(x[2]) => HMC(0.1, 5)),
            5;
            progress=false,
        )
        stat_names = filter(
            n -> occursin("acceptance_rate", string(n)), string.(collect(keys(chn3)))
        )
        @test length(stat_names) == 2
        @test !any(
            n -> occursin('[', n) || occursin(']', n) || occursin('.', n), stat_names
        )
    end

    @testset "component samplers keep their own init strategy" begin
        # Each variable is initialised by the strategy of the component that samples it, so
        # HMC's `InitFromUniform(-2, 2)` still applies to `h` inside Gibbs. `Beta(1, 1)` is
        # linked by the logit, so that range is `logistic(-2) < h < logistic(2)`.
        @model function beta_beta()
            h ~ Beta(1, 1)
            return m ~ Beta(1, 1)
        end
        lo, hi = inv(1 + exp(2)), inv(1 + exp(-2))
        # One sample per chain, so every draw is a freshly initialised one.
        chn = sample(
            StableRNG(468),
            beta_beta(),
            Gibbs(@varname(h) => HMC(0.1, 5), @varname(m) => MH()),
            MCMCThreads(),
            1,
            100;
            progress=false,
        )
        @test all(h -> lo < h < hi, vec(chn[@varname(h)]))
        # `m` is sampled by MH, which initialises from the prior, so it is not restricted to
        # the range HMC would use.
        @test any(m -> !(lo < m < hi), vec(chn[@varname(m)]))
    end

    @testset "a component may move another block's support, at the caller's risk" begin
        # None of these is refused. `absorb`, `discrete_support` and `discrete_vector` have
        # DISJOINT supports and are absorbed when split, giving a wrong answer with nothing
        # raised; `bounded` has NESTED supports and is correct either way. `disjoint_supports`
        # below measures the cost and the Turing.jl#2801 testset the safe case. Why this is
        # permitted is in the `Gibbs` docstring.
        @model function absorb()
            b ~ Bernoulli(0.5)
            if b == 1
                x ~ Dirichlet(3, 1.0)
            else
                x ~ MvNormal(zeros(3), LinearAlgebra.I)
            end
            return 0.5 ~ Normal(sum(x), 1.0)
        end
        @model function bounded()
            a ~ Uniform(-1.0, 0.0)
            x ~ truncated(Normal(0, 1); lower=a)
            return 0.3 ~ Normal(x, 0.5)
        end
        # A discrete variable is never linked, so its bijector is `identity` whatever its
        # support, and only the bounds separated these two branches while the rule existed.
        # Kept as a partition that must not be refused. Without
        # them the partition was accepted and absorbed, returning P(b=1) of 0.0 or 1.0 by
        # seed against an exact 0.5.
        @model function discrete_support()
            b ~ Bernoulli(0.5)
            if b == 1
                k ~ DiscreteUniform(1, 2)
            else
                k ~ DiscreteUniform(3, 4)
            end
            return 0.2 ~ Normal(k, 1.0)
        end
        # A discrete VECTOR has an `identity` bijector too, and restricting the bounds to
        # univariate distributions left it with nothing carrying its support at all -- the same
        # absorption one shape up, P(b=1) of 0.0 or 1.0 by seed against an exact 0.99999979.
        @model function discrete_vector()
            b ~ Bernoulli(0.5)
            if b == 1
                k ~ product_distribution([DiscreteUniform(1, 2), DiscreteUniform(1, 2)])
            else
                k ~ product_distribution([DiscreteUniform(3, 4), DiscreteUniform(3, 4)])
            end
            return 0.2 ~ Normal(sum(k), 1.0)
        end
        for (model, decider, affected) in (
            (absorb(), @varname(b), @varname(x)),
            (bounded(), @varname(a), @varname(x)),
            (discrete_support(), @varname(b), @varname(k)),
            (discrete_vector(), @varname(b), @varname(k)),
        )
            # Neither form is refused. For the three disjoint-support models the split answer
            # is wrong and the shared one right; for `bounded` both are right. Only the
            # absence of a refusal is asserted here -- the numbers are measured by
            # `disjoint_supports` below and by the #2801 testset.
            @test sample(
                Xoshiro(2),
                model,
                Gibbs(decider => MH(), affected => MH()),
                200;
                check_model=false,
                progress=false,
            ) isa Any
            @test sample(
                Xoshiro(2),
                model,
                Gibbs((decider, affected) => MH()),
                200;
                check_model=false,
                progress=false,
            ) isa Any
        end

        # A signature has to compare by support, not by object identity. `ProductBijector`
        # defines no `==`, so two bijectors built from the SAME distribution were unequal and a
        # partition that never changes anything was refused on its first sweep.
        @model function prodmodel()
            m ~ Normal(0, 1)
            x ~ product_distribution([Dirichlet(ones(3)), Dirichlet(ones(3))])
            return 0.2 ~ Normal(sum(x) + m, 1.0)
        end
        @test sample(
            Xoshiro(3),
            prodmodel(),
            Gibbs(@varname(m) => MH(), @varname(x) => MH()),
            100;
            check_model=false,
            progress=false,
        ) isa Any

        # A component may name what the model writes in one tilde by its elements. Asking
        # subsumption one way only made the ownership test refuse the very arrangement this
        # docstring prescribes, and left `block_fingerprint`'s layout empty so a relinking
        # passed the gate.
        @model function decided()
            a ~ Normal()
            if a > 0
                x ~ Dirichlet(3, 1.0)
            else
                x ~ MvNormal(zeros(3), LinearAlgebra.I)
            end
            return 0.4 ~ Normal(sum(x), 1.0)
        end
        elementwise_x = (@varname(x[1]), @varname(x[2]), @varname(x[3]))
        @test sample(
            Xoshiro(8),
            decided(),
            Gibbs((@varname(a), elementwise_x...) => MH()),
            100;
            check_model=false,
            progress=false,
        ) isa Any
        @model function relinked_by_b(y)
            b ~ Bernoulli(0.5)
            if b
                x ~ Dirichlet(3, 1.0)
            else
                x ~ MvNormal(zeros(3), LinearAlgebra.I)
            end
            return y ~ Normal(sum(x), 1.0)
        end
        @test_throws "does not implement" sample(
            Xoshiro(8),
            relinked_by_b(1.0),
            Gibbs((@varname(b), @varname(x)) => PG(10), elementwise_x => NUTS()),
            100;
            progress=false,
        )

        # The cost of the relaxation, measured. `Uniform(0,1)` against `Uniform(2,3)` has one
        # dimension either way and the same family, so nothing about it is refused; the split
        # chain is absorbed while the shared block is right.
        @model function disjoint_supports()
            b ~ Bernoulli(0.5)
            if b
                x ~ Uniform(0.0, 1.0)
            else
                x ~ Uniform(2.0, 3.0)
            end
            return 0.5 ~ Normal(x, 5.0)
        end
        shared = mean(
            sample(
                StableRNG(1),
                disjoint_supports(),
                Gibbs((@varname(b), @varname(x)) => MH()),
                3000;
                discard_initial=500,
                progress=false,
            )[@varname(b)],
        )
        @test shared ≈ 0.5 atol = 0.06
        split_chain = sample(
            StableRNG(1),
            disjoint_supports(),
            Gibbs(@varname(b) => MH(), @varname(x) => MH()),
            3000;
            discard_initial=500,
            check_model=false,
            progress=false,
        )
        # Absorbed: every draw of `b` is the value it started at. Asserted so that a future
        # change which makes this partition mix, or refuses it again, is noticed here.
        @test length(unique(vec(split_chain[@varname(b)]))) == 1

        # A distribution whose PARAMETERS depend on another block is the ordinary hierarchical
        # case and must not be refused: `x`'s support is the whole line every sweep.
        @model function meanshift()
            m ~ Normal(0, 1)
            x ~ Normal(m, 1)
            return 0.4 ~ Normal(x, 0.5)
        end
        for second in (MH(), NUTS())
            @test sample(
                Xoshiro(2),
                meanshift(),
                Gibbs(@varname(m) => second, @varname(x) => MH()),
                200;
                check_model=false,
                progress=false,
            ) isa Any
        end
    end

    @testset "an adaptive sampler with adaptation switched off works as a component" begin
        # `n_adapts = 0` asks `NUTS`/`HMCDA` not to adapt, which `AHMCAdaptor` honours with
        # `NoAdaptation`. Gibbs then has no preconditioner to renew a metric from, and every
        # such component used to fail on its first state update with
        # `FieldError: NoAdaptation has no field pc` -- while the same sampler used on its own
        # worked, which is what Turing.jl#2400 pins.
        @model function g()
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            return 1.5 ~ Normal(m, sqrt(s))
        end
        for component in (NUTS(0, 0.65), HMCDA(0, 0.65, 0.3))
            chn = sample(
                Xoshiro(1),
                g(),
                Gibbs(@varname(s) => MH(), @varname(m) => component),
                400;
                progress=false,
            )
            ms = vec(chn[@varname(m)])
            @test all(isfinite, ms)
            # Not adapting is fine; not moving is not.
            @test std(ms) > 0.1
        end
    end

    @testset "a variable appearing mid-run needs a suitable component sampler (#2810)" begin
        # The example from #2810: `z` exists only while `x > 0`, so it appears part-way through
        # a run. Whichever block owns it has to be able to sample a set of variables that
        # changes between sweeps.
        @model function f()
            x ~ Normal()
            y ~ Normal()
            if x > 0
                z ~ Normal()
            end
        end
        # Unassigned, `z` would be drawn once by whichever block reached it and then
        # conditioned on for the rest of the run.
        @test_throws ArgumentError sample(
            Xoshiro(470),
            f(),
            Gibbs(@varname(x) => MH(), @varname(y) => MH()),
            100;
            check_model=false,
            progress=false,
        )
        # Naming the components by sampler type made this message useless whenever two
        # components share one, which is the common case: it read "during a step of MH, which
        # does not sample it: MH does".
        #
        # `Xoshiro(5)` draws `x > 0`, so `z` is in the first snapshot, its component
        # initialises normally, and it is `x`'s step that later removes `z` -- the ownership
        # guard, which names both components.
        split_z = Gibbs(@varname(x) => MH(), @varname(y) => MH(), @varname(z) => MH())
        err = try
            sample(Xoshiro(5), f(), split_z, 200; check_model=false, progress=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("component sampling x", err.msg)
        @test occursin("component sampling z", err.msg)

        # `Xoshiro(470)` draws `x < 0`, so `z` is absent from the first snapshot and its
        # component has nothing to sample. That is refused before any component steps, which
        # unlike the case above does not depend on what the first sweep happens to draw. The
        # remedy is the same; the message cannot name `x`, because nothing tells Gibbs which
        # variable decides `z`'s existence.
        @test_throws "has nothing to sample" sample(
            Xoshiro(470), f(), split_z, 200; check_model=false, progress=false
        )

        # Seed 4's initial draw already takes the branch, so `z` is in the first snapshot and
        # the every-variable-has-a-component check refuses it at the initial step, before any
        # component has stepped -- a different guard from the seeds above, where `z` turns up
        # mid-run. Pinning the message keeps the two apart.
        err = try
            sample(
                Xoshiro(4),
                f(),
                Gibbs(@varname(x) => MH(), @varname(y) => MH()),
                100;
                check_model=false,
                progress=false,
            )
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("has no component for z", err.msg)

        # `x` and `z` share a block here, and a default `MH()` proposes both from their
        # priors. The proposal density then cancels against the prior and the acceptance ratio
        # collapses to a likelihood ratio, which is defined between spaces of different
        # dimension -- so this is a legitimate move across supports, not something to refuse.
        # The model is prior-only, so P(x > 0) = 1/2 exactly.
        ps = map((1, 4, 5)) do seed
            chn = sample(
                Xoshiro(seed),
                f(),
                Gibbs(@varname(y) => MH(), (@varname(x), @varname(z)) => MH()),
                20_000;
                discard_initial=5_000,
                check_model=false,
                progress=false,
            )
            return mean(vec(chn[@varname(x)]) .> 0)
        end
        @test mean(ps) ≈ 0.5 atol = 0.02

        # `MH` proposes every variable from its prior, so `q` cancels against `p` in the
        # acceptance ratio and a crossing is legitimate whatever the dimension. Per-variable
        # proposals, which would break that cancellation, no longer exist.
        @test Turing.Inference.allow_varying_dimension(MH())
        # The crossing variable need not be scalar.
        @model function vec_crossing()
            b ~ Bernoulli(0.5)
            mu = 0.0
            if b
                x ~ MvNormal(zeros(2), LinearAlgebra.I)
                mu = sum(x)
            end
            return 1.0 ~ Normal(mu, 1.0)
        end
        @test sample(
            StableRNG(6),
            vec_crossing(),
            Gibbs((@varname(b), @varname(x)) => MH()),
            200;
            check_model=false,
            progress=false,
        ) isa Any
        # Refused for ownership, not for the ratio: `x` decides whether `z` exists and sits
        # in another block.
        @test_throws ArgumentError sample(
            Xoshiro(470),
            f(),
            Gibbs(@varname(x) => MH(), @varname(y) => MH(), @varname(z) => MH()),
            100;
            check_model=false,
            progress=false,
        )
        # `PG` redraws whatever the model reaches, so it can own `z`. `x` decides whether `z`
        # exists, so it has to be in the same block: with `x` in a block of its own, that
        # component conditions on `z` while proposing an `x` for which `z` does not exist, and
        # the chain samples but comes back biased. The model is prior-only, so P(x > 0) = 1/2
        # measures whether the sweep is valid -- the split partition gives about 1/20.
        chn = sample(
            Xoshiro(470),
            f(),
            Gibbs(@varname(y) => MH(), (@varname(x), @varname(z)) => PG(20)),
            2000;
            check_model=false,
            progress=false,
        )
        @test size(chn, 1) == 2000
        @test any(!ismissing, chn[@varname(z)])
        xs = vec(chn[@varname(x)])
        @test count(>(0), xs) / length(xs) ≈ 0.5 atol = 0.06

        # Every variable has to be declared, so leaving `z` out is rejected whether it turns
        # up mid-run or is already there in the initialising draw.
        for seed in (3, 4)
            @test_throws ArgumentError sample(
                Xoshiro(seed),
                f(),
                Gibbs(@varname(x) => PG(20), @varname(y) => MH()),
                200;
                check_model=false,
                progress=false,
            )
        end

        # A new element counts too, not just a new name.
        @model function growing()
            n ~ Bernoulli(0.5)
            x = Vector{Float64}(undef, 2)
            x[1] ~ Normal()
            if n == 1
                x[2] ~ Normal()
            end
        end
        @test_throws ArgumentError sample(
            Xoshiro(4),
            growing(),
            Gibbs(@varname(n) => MH(), @varname(x[1]) => MH()),
            40;
            check_model=false,
            progress=false,
        )
    end

    @testset "a `:=` quantity is not a variable that appeared" begin
        # `y` is not sampled, so it must never be taken for a variable that turned up mid-run
        # and needs a component. Which seeds reach the branch on the initial draw decides
        # whether the mistake fires, so check several.
        @model function colon_eq_branch()
            x ~ Normal()
            if x > 0
                y := x^2
            end
            return 1.0 ~ Normal(x, 1.0)
        end
        for seed in 1:5
            @test sample(
                Xoshiro(seed),
                colon_eq_branch(),
                Gibbs(@varname(x) => MH()),
                60;
                check_model=false,
                progress=false,
            ) isa Any
        end

        # Excluded from what components report to Gibbs, but still rebuilt for the chain.
        @model function colon_eq_plain()
            m ~ Normal()
            n := m + 100
            return 0.0 ~ Normal(m)
        end
        chn = sample(
            Xoshiro(1), colon_eq_plain(), Gibbs(@varname(m) => MH()), 10; progress=false
        )
        @test all(
            ≈(100), collect(skipmissing(vec(chn[@varname(n)]))) .- vec(chn[@varname(m)])
        )
    end

    @testset "an array whose length is itself sampled" begin
        @model function varying_length()
            n ~ Categorical([1 / 3, 1 / 3, 1 / 3])
            x = Vector{Float64}(undef, n)
            for i in 1:n
                x[i] ~ Normal()
            end
            return 1.0 ~ Normal(sum(x), 1.0)
        end
        # `n` decides how many `x[i]` exist, so it has to share their block. The snapshot then
        # merges an `x` of one length with an `x` of another, which needs DynamicPPL 0.42.8.
        chn = sample(
            Xoshiro(100),
            varying_length(),
            Gibbs((@varname(n), @varname(x)) => PG(10)),
            200;
            check_model=false,
            progress=false,
        )
        @test length(unique(vec(chn[@varname(n)]))) == 3

        # Split off, `x[2]` comes and goes under the `n` component's step while the `x`
        # component is the one that samples it, so the partition is rejected before `PG`'s own
        # "the number of observations must not be random" guard is reached.
        @test_throws ArgumentError sample(
            Xoshiro(100),
            varying_length(),
            Gibbs(@varname(n) => PG(10), @varname(x) => PG(10)),
            50;
            check_model=false,
            progress=false,
        )
    end

    @testset "a component may not own part of a stored variable" begin
        @model function pair()
            x = Vector{Float64}(undef, 2)
            x[1] ~ Normal()
            x[2] ~ Normal()
            return 0.0 ~ Normal(x[1] + x[2], 0.1)
        end
        # Whether the values store `x` as a unit is what decides this, and that is a property
        # of the model's tilde statements rather than of the samplers: the snapshot comes from
        # a model evaluation, so `x[1] ~` and `x[2] ~` give a key each and the first component
        # can be handed `x[2]` alone whatever holds the containing block.
        for component in (MH(), HMC(0.05, 3))
            spl = Gibbs(@varname(x[1]) => MH(), @varname(x) => component)
            @test sample(StableRNG(468), pair(), spl, 5; progress=false) isa Any
        end

        # A single tilde over the whole vector cannot be split, whatever samples it.
        @model function joint()
            x ~ MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])
            return 0.0 ~ Normal(x[1] + x[2], 0.1)
        end
        spl = Gibbs(@varname(x[1]) => MH(), @varname(x) => MH())
        @test_throws ArgumentError sample(StableRNG(468), joint(), spl, 5; progress=false)

        # Claiming every part separately is still unsplittable, and says so rather than
        # reporting `x` as a variable nobody claimed.
        spl = Gibbs(@varname(x[1]) => MH(), @varname(x[2]) => MH())
        err = try
            sample(StableRNG(468), joint(), spl, 5; progress=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("claim parts of x separately", err.msg)
        # Element-wise tildes give a key each, so the same partition is expressible.
        @test sample(StableRNG(468), pair(), spl, 5; progress=false) isa Any

        # One component owning every leaf is not a split at all: it can be handed the whole
        # value, so this must sample even though no declared varname subsumes the `x` key.
        spl = Gibbs((@varname(x[1]), @varname(x[2])) => MH())
        @test sample(StableRNG(468), joint(), spl, 5; progress=false) isa Any
    end

    @testset "non-identity varnames" begin
        struct Wrap{T}
            a::T
        end
        @model function model1((::Type{T})=Float64) where {T}
            x = Vector{T}(undef, 1)
            x[1] ~ Normal()
            y = Wrap{T}(0.0)
            return y.a ~ Normal()
        end
        model = model1()
        spl = Gibbs(@varname(x[1]) => HMC(0.5, 10), @varname(y.a) => MH())
        @test sample(model, spl, 10) isa VNChain
        spl = Gibbs((@varname(x[1]), @varname(y.a)) => HMC(0.5, 10))
        @test sample(model, spl, 10) isa VNChain
    end

    @testset "submodels" begin
        @model inner() = x ~ Normal()
        @model function outer()
            a ~ to_submodel(inner())
            _ignored ~ to_submodel(prefix(inner(), @varname(b)), false)
            return _also_ignored ~ to_submodel(inner(), false)
        end
        model = outer()
        spl = Gibbs(
            @varname(a.x) => HMC(0.5, 10), @varname(b.x) => MH(), @varname(x) => MH()
        )
        @test sample(model, spl, 10) isa VNChain
        spl = Gibbs((@varname(a.x), @varname(b.x), @varname(x)) => MH())
        @test sample(model, spl, 10) isa VNChain

        @testset "regression test for #2798" begin
            @model function vect_inner()
                my_vec = zeros(2)
                my_vec[1] ~ Normal()
                return my_vec
            end
            @model function vect_middle()
                x ~ to_submodel(vect_inner())
                return y ~ Normal(x[1], 1)
            end
            @model function vect_outer()
                a ~ to_submodel(vect_middle())
                return z ~ Normal(a)
            end
            sample(vect_middle(), Gibbs(@varname(x) => MH(), @varname(y) => MH()), 10)
            sample(vect_outer(), Gibbs(@varname(a) => MH(), @varname(z) => MH()), 10)
        end
    end

    @testset "CSMC + ESS" begin
        model = MoGtest_default
        spl = Gibbs(
            (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
            @varname(mu1) => ESS(),
            @varname(mu2) => ESS(),
        )
        vns = (
            @varname(z1),
            @varname(z2),
            @varname(z3),
            @varname(z4),
            @varname(mu1),
            @varname(mu2)
        )
        # `step`
        rng = Random.default_rng()
        transition, state = AbstractMCMC.step(rng, model, spl)
        check_transition_varnames(transition, vns)
        for _ in 1:5
            transition, state = AbstractMCMC.step(rng, model, spl, state)
            check_transition_varnames(transition, vns)
        end

        # Sample!
        chain = sample(StableRNG(42), MoGtest_default, spl, 1000; progress=false)
        check_MoGtest_default(chain; atol=0.2)
    end

    @testset "CSMC + ESS (usage of implicit varname)" begin
        model = MoGtest_default_z_vector
        spl = Gibbs(@varname(z) => CSMC(15), @varname(mu1) => ESS(), @varname(mu2) => ESS())
        vns = (@varname(z), @varname(mu1), @varname(mu2))
        # `step`
        rng = Random.default_rng()
        transition, state = AbstractMCMC.step(rng, model, spl)
        check_transition_varnames(transition, vns)
        for _ in 1:5
            transition, state = AbstractMCMC.step(rng, model, spl, state)
            check_transition_varnames(transition, vns)
        end

        # Sample!
        chain = sample(StableRNG(42), model, spl, 1000; progress=false)
        check_MoGtest_default_z_vector(chain; atol=0.2)
    end

    @testset "externalsampler" begin
        function check_logp_correct(sampler)
            @testset "logp is set correctly" begin
                @model logp_check() = x ~ Normal()
                chn = sample(
                    logp_check(), Gibbs(@varname(x) => sampler), 100; progress=false
                )
                @test isapprox(logpdf.(Normal(), chn[@varname(x)]), chn[:logjoint])
            end
        end

        @model function demo_gibbs_external()
            m1 ~ Normal()
            m2 ~ Normal()

            -1 ~ Normal(m1, 1)
            +1 ~ Normal(m1 + m2, 1)

            return (; m1, m2)
        end

        model = demo_gibbs_external()
        samplers_inner = [
            externalsampler(AdvancedMH.RWMH(1)),
            externalsampler(AdvancedHMC.HMC(1e-1, 32); adtype=AutoForwardDiff()),
            externalsampler(AdvancedHMC.HMC(1e-1, 32); adtype=AutoReverseDiff()),
            externalsampler(
                AdvancedHMC.HMC(1e-1, 32); adtype=AutoReverseDiff(; compile=true)
            ),
        ]
        @testset "$(sampler_inner)" for sampler_inner in samplers_inner
            sampler = Gibbs(@varname(m1) => sampler_inner, @varname(m2) => sampler_inner)
            chain = sample(
                StableRNG(42),
                model,
                sampler,
                1000;
                discard_initial=1000,
                thinning=10,
                n_adapts=0,
            )
            check_numerical(chain, [:m1, :m2], [-0.2, 0.6]; atol=0.1)
            check_logp_correct(sampler_inner)
        end

        # Gibbs drops component transitions, so a component's statistics have to be read off
        # its state. An external sampler's were dropped silently, since `TuringState` had no
        # `gibbs_get_stats` and fell through to the empty default.
        spl = externalsampler(AdvancedHMC.HMC(1e-1, 32); adtype=AutoForwardDiff())
        chain = sample(
            StableRNG(42),
            model,
            Gibbs(@varname(m1) => spl, @varname(m2) => spl),
            20;
            n_adapts=0,
            progress=false,
        )
        for name in ("m1_acceptance_rate", "m2_acceptance_rate")
            @test !isempty(collect(skipmissing(vec(chain[Symbol(name)]))))
        end
    end

    # Test a model that where the sampler needs to link a variable, which consequently
    # changes dimension. This used to error because the initial value `VarInfo`,
    # obtained from just `VarInfo(model)`, had a value of dimension 2 for `w`, and the one
    # coming out of the initial step of the component sampler had a dimension of 1, since
    # the latter was linked. `merge` of the varinfos couldn't handle that before DPPL
    # 0.34.1.
    @testset "linking changes dimension" begin
        @model function dirichlet_model()
            K = 2
            w ~ Dirichlet(K, 1.0)
            for i in 1:K
                0.1 ~ Normal(w[i], 1.0)
            end
        end

        model = dirichlet_model()
        sampler = Gibbs(:w => HMC(0.05, 10))
        @test (sample(model, sampler, 10); true)
    end

    @testset "dynamic transformations with linked samplers" begin
        # Turing.jl#2801. `y`'s transform depends on `x`, and the linked value for `y` was
        # once not updated when `x` changed, which gave wrong answers.
        #
        # `x` moves the support of a `y` in another block, which Gibbs permits: the supports
        # are nested, every reachable pair overlaps, and the split partition is irreducible.
        # This is the safe side of the warning in the `Gibbs` docstring, and the ground truth
        # below is what makes it safe rather than merely unrefused -- contrast
        # `disjoint_supports` above, which is permitted and wrong.
        @model function dyn()
            x ~ Uniform(-5, 5)
            return y ~ truncated(Normal(); lower=x)
        end
        model = dyn()

        samplers = (
            Gibbs(:x => MH(), :y => HMC(0.1, 20)), Gibbs(:x => MH(), :y => MH([1.0;;]))
        )
        for spl in samplers
            chn = sample(
                StableRNG(468), model, spl, MCMCThreads(), 100000, 4; verbose=false
            )
            # Ground truth from NUTS. Asserted on the SPLIT partitions, which are the only
            # ones that condition `y` on an `x` sampled elsewhere: a range check on `x`, or
            # a mean over one block holding both, passes with the staleness reintroduced.
            @test mean(chn[:x]) ≈ 0.0 atol = 0.1
            @test mean(chn[:y]) ≈ 1.5 atol = 0.1
        end
    end
end

end
