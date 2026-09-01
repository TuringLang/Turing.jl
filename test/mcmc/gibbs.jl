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
using Distributions: InverseGamma, Normal
using Distributions: sample
using DynamicPPL: DynamicPPL
using FlexiChains: FlexiChains
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_broken, @test_throws, @testset
using Turing
using Turing: Inference
using Turing.Inference: AdvancedHMC, AdvancedMH

const TuringDistributionsExt = Base.get_extension(Turing, :TuringDistributionsExt)
using .TuringDistributionsExt: ChineseRestaurantProcess, DirichletProcess

function check_transition_varnames(transition::DynamicPPL.ParamsWithStats, parent_varnames)
    for vn in keys(transition.params)
        @test any(Base.Fix2(DynamicPPL.subsumes, vn), parent_varnames)
    end
end

@testset verbose = true "Gibbs conditioning" begin
    @testset "type stability" begin
        struct Wrapper{T<:Real}
            a::T
        end

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

@testset "latent declared as a missing model argument" begin
    # Conditioning has to reach a variable that is also a model argument: Gibbs conditions
    # every non-target variable, and here the non-target `x` is an argument bound to
    # `missing`. Needs DynamicPPL to let `condition` take precedence over a missing argument
    # (DynamicPPL.jl#1457); otherwise `missing` is passed to the likelihood and errors.
    @model function impute(x)
        m ~ Normal(0, 1)
        x ~ Normal(m, 1)
        return 2.0 ~ Normal(x, 1)
    end
    chn = sample(
        StableRNG(468),
        impute(missing),
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

    struct Wrapper{T<:Real}
        a::T
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

    # we need some state type to implement the Gibbs interface (we can't just use `nothing`)
    struct TrivialState end
    Turing.Inference.gibbs_get_parameter_values(::TrivialState) = VarNamedTuple()
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
        return nothing, TrivialState()
    end

    function AbstractMCMC.step_warmup(
        ::Random.AbstractRNG,
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
        return nothing, TrivialState()
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

        # A Metropolis-Hastings ratio between states of different dimension is not the one the
        # algorithm assumes, and a `LogDensityFunction` fixes its layout at the first step, so
        # both are rejected rather than left to sample a different target than the model's.
        @test_throws ArgumentError sample(
            StableRNG(42), model, Gibbs(:b => MH(), :θ => MH()), 200; progress=false
        )
        @test_throws ArgumentError sample(
            StableRNG(42), model, Gibbs(:b => MH(), :θ => HMC(0.1, 5)), 200; progress=false
        )

        # `PG` rebuilds its trace each sweep, so it can own the block whose shape varies.
        chn = sample(
            StableRNG(42),
            model,
            Gibbs(:b => MH(), :θ => PG(50)),
            2000;
            discard_initial=1000,
            progress=false,
        )
        @test size(chn, 1) == 2000

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
        # Seed 4's initial draw already takes the branch, so `z` reaches the snapshot before
        # any component steps. Nothing then sees it appear, and it stayed frozen for the run.
        @test_throws ArgumentError sample(
            Xoshiro(4),
            f(),
            Gibbs(@varname(x) => MH(), @varname(y) => MH()),
            100;
            check_model=false,
            progress=false,
        )
        # Assigned to MH, whose acceptance ratio is not valid across a change of dimension.
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

        # Undeclared, `z` is taken on by the component that reaches it, which then keeps
        # sampling it instead of conditioning on one draw. `x` decides whether `z` exists and
        # is in that same block, so the sweep is valid: P(x > 0) = 1/2 on this prior-only
        # model.
        chn = sample(
            Xoshiro(3),
            f(),
            Gibbs(@varname(x) => PG(20), @varname(y) => MH()),
            2000;
            check_model=false,
            progress=false,
        )
        zs = collect(skipmissing(vec(chn[@varname(z)])))
        @test length(unique(zs)) > 1
        xs = vec(chn[@varname(x)])
        @test count(>(0), xs) / length(xs) ≈ 0.5 atol = 0.06

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

        # Split off, the conditioned `x[i]` become observations whose number depends on `n`,
        # and `PG` says so rather than sampling something wrong.
        @test_throws ErrorException sample(
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
        # One component owns `x[1]` while another owns all of `x`, so conditioning the first
        # would have to cover part of `x` and leave the rest free. The values cannot express
        # that, and dropping `x` wholesale would silently let the first component sample
        # `x[2]` as well, so this must fail rather than sample the wrong block.
        spl = Gibbs(@varname(x[1]) => MH(), @varname(x) => HMC(0.05, 3))
        @test_throws ArgumentError sample(StableRNG(468), pair(), spl, 5; progress=false)

        # Whether the values store `x` as a unit is what decides this, not the declared
        # varnames. The same partition with MH on the containing block is expressible, because
        # MH reports a key per element and the first component can be given `x[2]` alone.
        spl = Gibbs(@varname(x[1]) => MH(), @varname(x) => MH())
        @test sample(StableRNG(468), pair(), spl, 5; progress=false) isa Any

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
        # See https://github.com/TuringLang/Turing.jl/issues/2801.
        # The issue there was that the linked value for `y` was never updated when `x`
        # changed, even though it should (the transform for `y` depends on `x`), leading to
        # incorrect results.
        @model function dyn()
            x ~ Uniform(-5, 5)
            return y ~ truncated(Normal(); lower=x)
        end
        model = dyn()

        for spl in (
            Gibbs(:x => MH(), :y => HMC(0.1, 20)),
            Gibbs(:x => MH(), :y => MH(:y => LinkedRW(1.0))),
        )
            chn = sample(
                StableRNG(468), model, spl, MCMCThreads(), 100000, 4; verbose=false
            )
            # ground truth obtained from NUTS
            @test mean(chn[:x]) ≈ 0.0 atol = 0.1
            @test mean(chn[:y]) ≈ 1.5 atol = 0.1
        end
    end
end

end
