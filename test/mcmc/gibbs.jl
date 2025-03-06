module GibbsTests

using ..Models: MoGtest_default, MoGtest_default_z_vector, gdemo, gdemo_default
using ..NumericalTests:
    check_MoGtest_default,
    check_MoGtest_default_z_vector,
    check_gdemo,
    check_numerical,
    two_sample_test
import ..ADUtils
import Combinatorics
using Distributions: InverseGamma, Normal
using Distributions: sample
using DynamicPPL: DynamicPPL
using ForwardDiff: ForwardDiff
using Random: Random
using ReverseDiff: ReverseDiff
import Mooncake
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_broken, @test_deprecated, @test_throws, @testset
using Turing
using Turing: Inference
using Turing.Inference: AdvancedHMC, AdvancedMH
using Turing.RandomMeasures: ChineseRestaurantProcess, DirichletProcess

function check_transition_varnames(transition::Turing.Inference.Transition, parent_varnames)
    transition_varnames = mapreduce(vcat, transition.θ) do vn_and_val
        [first(vn_and_val)]
    end
    # Varnames in `transition` should be subsumed by those in `parent_varnames`.
    for vn in transition_varnames
        @test any(Base.Fix2(DynamicPPL.subsumes, vn), parent_varnames)
    end
end

const DEMO_MODELS_WITHOUT_DOT_ASSUME = Union{
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_index_observe)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_multivariate_observe)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_dot_observe)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_multivariate_observe_literal)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_observe_literal)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_dot_observe_literal)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix)},
}
has_dot_assume(::DEMO_MODELS_WITHOUT_DOT_ASSUME) = false
has_dot_assume(::DynamicPPL.Model) = true

@testset "GibbsContext" begin
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
            obs1 ~ Normal(s, 1)
            obs2 ~ Poisson(y[3])
            return obs1, obs2, variance, z, y, s
        end

        model = test_model(1.2, 2, 10, 2.5)
        all_varnames = DynamicPPL.VarName[@varname(variance), @varname(z), @varname(y)]
        # All combinations of elements in all_varnames.
        target_vn_combinations = Iterators.flatten(
            Iterators.map(
                n -> Combinatorics.combinations(all_varnames, n), 1:length(all_varnames)
            ),
        )

        @testset "$(target_vns)" for target_vns in target_vn_combinations
            global_varinfo = DynamicPPL.VarInfo(model)
            target_vns = collect(target_vns)
            local_varinfo = DynamicPPL.subset(global_varinfo, target_vns)
            ctx = Turing.Inference.GibbsContext(
                target_vns, Ref(global_varinfo), Turing.DefaultContext()
            )

            # Check that the correct varnames are conditioned, and that getting their
            # values is type stable when the varinfo is.
            for k in keys(global_varinfo)
                is_target = any(Iterators.map(vn -> DynamicPPL.subsumes(vn, k), target_vns))
                @test Turing.Inference.is_target_varname(ctx, k) == is_target
                if !is_target
                    @inferred Turing.Inference.get_conditioned_gibbs(ctx, k)
                end
            end

            # Check the type stability also in the dot_tilde pipeline.
            for k in all_varnames
                # The map(identity, ...) part is there to concretise the eltype.
                subkeys = map(
                    identity, filter(vn -> DynamicPPL.subsumes(k, vn), keys(global_varinfo))
                )
                is_target = (k in target_vns)
                @test Turing.Inference.is_target_varname(ctx, subkeys) == is_target
                if !is_target
                    @inferred Turing.Inference.get_conditioned_gibbs(ctx, subkeys)
                end
            end

            # Check that evaluate!! and the result it returns are type stable.
            conditioned_model = DynamicPPL.contextualize(model, ctx)
            _, post_eval_varinfo = @inferred DynamicPPL.evaluate!!(
                conditioned_model, local_varinfo
            )
            for k in keys(post_eval_varinfo)
                @inferred post_eval_varinfo[k]
            end
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
    @test_throws ArgumentError Gibbs(@varname(s) => IS())
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

# Test that the samplers are being called in the correct order, on the correct target
# variables.
@testset "Sampler call order" begin
    # A wrapper around inference algorithms to allow intercepting the dispatch cascade to
    # collect testing information.
    struct AlgWrapper{Alg<:Inference.InferenceAlgorithm} <: Inference.InferenceAlgorithm
        inner::Alg
    end

    unwrap_sampler(sampler::DynamicPPL.Sampler{<:AlgWrapper}) =
        DynamicPPL.Sampler(sampler.alg.inner, sampler.selector)

    # Methods we need to define to be able to use AlgWrapper instead of an actual algorithm.
    # They all just propagate the call to the inner algorithm.
    Inference.isgibbscomponent(wrap::AlgWrapper) = Inference.isgibbscomponent(wrap.inner)
    Inference.drop_space(wrap::AlgWrapper) = AlgWrapper(Inference.drop_space(wrap.inner))
    function Inference.setparams_varinfo!!(
        model::DynamicPPL.Model,
        sampler::DynamicPPL.Sampler{<:AlgWrapper},
        state,
        params::Turing.AbstractVarInfo,
    )
        return Inference.setparams_varinfo!!(model, unwrap_sampler(sampler), state, params)
    end

    function target_vns(::Inference.GibbsContext{VNs}) where {VNs}
        return VNs
    end

    # targets_and_algs will be a list of tuples, where the first element is the target_vns
    # of a component sampler, and the second element is the component sampler itself.
    # It is modified by the capture_targets_and_algs function.
    targets_and_algs = Any[]

    function capture_targets_and_algs(sampler, context)
        if DynamicPPL.NodeTrait(context) == DynamicPPL.IsLeaf()
            return nothing
        end
        if context isa Inference.GibbsContext
            push!(targets_and_algs, (target_vns(context), sampler))
        end
        return capture_targets_and_algs(sampler, DynamicPPL.childcontext(context))
    end

    # The methods that capture testing information for us.
    function Turing.AbstractMCMC.step(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        sampler::DynamicPPL.Sampler{<:AlgWrapper},
        args...;
        kwargs...,
    )
        capture_targets_and_algs(sampler.alg.inner, model.context)
        return Turing.AbstractMCMC.step(
            rng, model, unwrap_sampler(sampler), args...; kwargs...
        )
    end

    function Turing.DynamicPPL.initialstep(
        rng::Random.AbstractRNG,
        model::DynamicPPL.Model,
        sampler::DynamicPPL.Sampler{<:AlgWrapper},
        args...;
        kwargs...,
    )
        capture_targets_and_algs(sampler.alg.inner, model.context)
        return Turing.DynamicPPL.initialstep(
            rng, model, unwrap_sampler(sampler), args...; kwargs...
        )
    end

    # A test model that includes several different kinds of tilde syntax.
    @model function test_model(val, ::Type{M}=Vector{Float64}) where {M}
        s ~ Normal(0.1, 0.2)
        m ~ Poisson()
        val ~ Normal(s, 1)
        1.0 ~ Normal(s + m, 1)

        n := m + 1
        xs = M(undef, n)
        for i in eachindex(xs)
            xs[i] ~ Beta(0.5, 0.5)
        end

        ys = M(undef, 2)
        ys .~ Beta(1.0, 1.0)
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
        @varname(ys) => AlgWrapper(nuts),
        (@varname(xs), @varname(ys)) => AlgWrapper(hmc),
        @varname(s) => AlgWrapper(mh),
    )
    chain = sample(test_model(-1), sampler, 2)

    expected_targets_and_algs_per_iteration = [
        ((:s,), mh),
        ((:s, :m), mh),
        ((:m,), pg),
        ((:xs,), hmc),
        ((:ys,), nuts),
        ((:ys,), nuts),
        ((:xs, :ys), hmc),
        ((:s,), mh),
    ]
    @test targets_and_algs == vcat(
        expected_targets_and_algs_per_iteration, expected_targets_and_algs_per_iteration
    )
end

@testset "Equivalence of RepeatSampler and repeating Sampler" begin
    sampler1 = Gibbs(@varname(s) => RepeatSampler(MH(), 3), @varname(m) => ESS())
    sampler2 = Gibbs(
        @varname(s) => MH(), @varname(s) => MH(), @varname(s) => MH(), @varname(m) => ESS()
    )
    Random.seed!(23)
    chain1 = sample(gdemo_default, sampler1, 10)
    Random.seed!(23)
    chain2 = sample(gdemo_default, sampler1, 10)
    @test chain1.value == chain2.value
end

@testset "Gibbs warmup" begin
    # An inference algorithm, for testing purposes, that records how many warm-up steps
    # and how many non-warm-up steps haven been taken.
    mutable struct WarmupCounter <: Inference.InferenceAlgorithm
        warmup_init_count::Int
        non_warmup_init_count::Int
        warmup_count::Int
        non_warmup_count::Int

        WarmupCounter() = new(0, 0, 0, 0)
    end

    Turing.Inference.drop_space(wuc::WarmupCounter) = wuc
    Turing.Inference.getspace(::WarmupCounter) = ()
    Turing.Inference.isgibbscomponent(::WarmupCounter) = true

    # A trivial state that holds nothing but a VarInfo, to be used with WarmupCounter.
    struct VarInfoState{T}
        vi::T
    end

    Turing.Inference.varinfo(state::VarInfoState) = state.vi
    function Turing.Inference.setparams_varinfo!!(
        ::DynamicPPL.Model,
        ::DynamicPPL.Sampler,
        ::VarInfoState,
        params::DynamicPPL.AbstractVarInfo,
    )
        return VarInfoState(params)
    end

    function AbstractMCMC.step(
        ::Random.AbstractRNG,
        model::DynamicPPL.Model,
        spl::DynamicPPL.Sampler{<:WarmupCounter};
        kwargs...,
    )
        spl.alg.non_warmup_init_count += 1
        return Turing.Inference.Transition(nothing, 0.0),
        VarInfoState(DynamicPPL.VarInfo(model))
    end

    function AbstractMCMC.step_warmup(
        ::Random.AbstractRNG,
        model::DynamicPPL.Model,
        spl::DynamicPPL.Sampler{<:WarmupCounter};
        kwargs...,
    )
        spl.alg.warmup_init_count += 1
        return Turing.Inference.Transition(nothing, 0.0),
        VarInfoState(DynamicPPL.VarInfo(model))
    end

    function AbstractMCMC.step(
        ::Random.AbstractRNG,
        ::DynamicPPL.Model,
        spl::DynamicPPL.Sampler{<:WarmupCounter},
        s::VarInfoState;
        kwargs...,
    )
        spl.alg.non_warmup_count += 1
        return Turing.Inference.Transition(nothing, 0.0), s
    end

    function AbstractMCMC.step_warmup(
        ::Random.AbstractRNG,
        ::DynamicPPL.Model,
        spl::DynamicPPL.Sampler{<:WarmupCounter},
        s::VarInfoState;
        kwargs...,
    )
        spl.alg.warmup_count += 1
        return Turing.Inference.Transition(nothing, 0.0), s
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

@testset "Testing gibbs.jl with $adbackend" for adbackend in ADUtils.adbackends
    @info "Starting Gibbs tests with $adbackend"
    @testset "Deprecated Gibbs constructors" begin
        N = 10
        @test_deprecated s1 = Gibbs(HMC(0.1, 5, :s, :m; adtype=adbackend))
        @test_deprecated s2 = Gibbs(PG(10, :s, :m))
        @test_deprecated s3 = Gibbs(PG(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        @test_deprecated s4 = Gibbs(PG(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        @test_deprecated s5 = Gibbs(CSMC(3, :s), HMC(0.4, 8, :m; adtype=adbackend))
        @test_deprecated s6 = Gibbs(HMC(0.1, 5, :s; adtype=adbackend), ESS(:m))
        @test_deprecated s7 = Gibbs((HMC(0.1, 5, :s; adtype=adbackend), 2), (ESS(:m), 3))
        for s in (s1, s2, s3, s4, s5, s6, s7)
            @test DynamicPPL.alg_str(Turing.Sampler(s, gdemo_default)) == "Gibbs"
        end

        # Check that the samplers work despite using the deprecated constructor.
        sample(gdemo_default, s1, N)
        sample(gdemo_default, s2, N)
        sample(gdemo_default, s3, N)
        sample(gdemo_default, s4, N)
        sample(gdemo_default, s5, N)
        sample(gdemo_default, s6, N)
        sample(gdemo_default, s7, N)

        g = Turing.Sampler(s3, gdemo_default)
        @test sample(gdemo_default, g, N) isa MCMCChains.Chains
    end

    @testset "Gibbs constructors" begin
        # Create Gibbs samplers with various configurations and ways of passing the
        # arguments, and run them all on the `gdemo_default` model, see that nothing breaks.
        N = 10
        # Two variables being sampled by one sampler.
        s1 = Gibbs((@varname(s), @varname(m)) => HMC(0.1, 5; adtype=adbackend))
        s2 = Gibbs((@varname(s), :m) => PG(10))
        # As above but different samplers and using kwargs.
        s3 = Gibbs(:s => CSMC(3), :m => HMCDA(200, 0.65, 0.15; adtype=adbackend))
        s4 = Gibbs(@varname(s) => HMC(0.1, 5; adtype=adbackend), @varname(m) => ESS())
        # Multiple instnaces of the same sampler. This implements running, in this case,
        # 3 steps of HMC on m and 2 steps of PG on m in every iteration of Gibbs.
        s5 = begin
            hmc = HMC(0.1, 5; adtype=adbackend)
            pg = PG(10)
            vns = @varname(s)
            vnm = @varname(m)
            Gibbs(vns => hmc, vns => hmc, vns => hmc, vnm => pg, vnm => pg)
        end
        # Same thing but using RepeatSampler.
        s6 = Gibbs(
            @varname(s) => RepeatSampler(HMC(0.1, 5; adtype=adbackend), 3),
            @varname(m) => RepeatSampler(PG(10), 2),
        )
        for s in (s1, s2, s3, s4, s5, s6)
            @test DynamicPPL.alg_str(Turing.Sampler(s, gdemo_default)) == "Gibbs"
        end

        @test sample(gdemo_default, s1, N) isa MCMCChains.Chains
        @test sample(gdemo_default, s2, N) isa MCMCChains.Chains
        @test sample(gdemo_default, s3, N) isa MCMCChains.Chains
        @test sample(gdemo_default, s4, N) isa MCMCChains.Chains
        @test sample(gdemo_default, s5, N) isa MCMCChains.Chains
        @test sample(gdemo_default, s6, N) isa MCMCChains.Chains

        g = Turing.Sampler(s3, gdemo_default)
        @test sample(gdemo_default, g, N) isa MCMCChains.Chains
    end

    # Test various combinations of samplers against models for which we know the analytical
    # posterior mean.
    @testset "Gibbs inference" begin
        @testset "CSMC and HMC on gdemo" begin
            alg = Gibbs(:s => CSMC(15), :m => HMC(0.2, 4; adtype=adbackend))
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_numerical(chain, [:m], [7 / 6]; atol=0.15)
            # Be more relaxed with the tolerance of the variance.
            check_numerical(chain, [:s], [49 / 24]; atol=0.35)
        end

        @testset "MH and HMCDA on gdemo" begin
            alg = Gibbs(:s => MH(), :m => HMCDA(200, 0.65, 0.3; adtype=adbackend))
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)
        end

        @testset "CSMC and ESS on gdemo" begin
            alg = Gibbs(:s => CSMC(15), :m => ESS())
            chain = sample(gdemo(1.5, 2.0), alg, 3_000)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)
        end

        # TODO(mhauru) Why is this in the Gibbs test suite?
        @testset "CSMC on gdemo" begin
            alg = CSMC(15)
            chain = sample(gdemo(1.5, 2.0), alg, 4_000)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)
        end

        @testset "PG and HMC on MoGtest_default" begin
            gibbs = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => PG(15),
                (@varname(mu1), @varname(mu2)) => HMC(0.15, 3; adtype=adbackend),
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
                @varname(m) => HMC(0.2, 4; adtype=adbackend),
                (@varname(m), @varname(s)) => HMC(0.2, 4; adtype=adbackend),
            )
            chain = sample(gdemo(1.5, 2.0), alg, 500)
            check_gdemo(chain; atol=0.15)
        end

        @testset "Multiple overlapping samplers on MoGtest_default" begin
            gibbs = Gibbs(
                (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => PG(15),
                (@varname(z1), @varname(z2)) => PG(15),
                (@varname(mu1), @varname(mu2)) => HMC(0.15, 3; adtype=adbackend),
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
            samples::Vector,
            ::typeof(model),
            ::Turing.Sampler{<:Gibbs},
            state,
            ::Type{MCMCChains.Chains};
            kwargs...,
        )
            samples isa Vector{<:Inference.Transition} || error("incorrect transitions")
            return nothing
        end

        function callback(rng, model, sampler, sample, state, i; kwargs...)
            sample isa Inference.Transition || error("incorrect sample")
            return nothing
        end

        alg = Gibbs(:s => MH(), :m => HMC(0.2, 4; adtype=adbackend))
        sample(model, alg, 100; callback=callback)
    end

    @testset "dynamic model" begin
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
            for k in 1:Kmax
                m[k] ~ Normal(1.0, 1.0)
            end
        end
        num_zs = 100
        num_samples = 10_000
        model = imm(Random.randn(num_zs), 1.0)
        # https://github.com/TuringLang/Turing.jl/issues/1725
        # sample(model, Gibbs(:z => MH(), :m => HMC(0.01, 4)), 100);
        chn = sample(
            StableRNG(23),
            model,
            Gibbs(:z => PG(10), :m => HMC(0.01, 4; adtype=adbackend)),
            num_samples,
        )
        # The number of m variables that have a non-zero value in a sample.
        num_ms = count(ismissing.(Array(chn[:, (num_zs + 1):end, 1])); dims=2)
        # The below are regression tests. The values we are comparing against are from
        # running the above model on the "old" Gibbs sampler that was in place still on
        # 2024-11-20. The model was run 5 times with 10_000 samples each time. The values
        # to compare to are the mean of those 5 runs, atol is roughly estimated from the
        # standard deviation of those 5 runs.
        # TODO(mhauru) Could we do something smarter here? Maybe a dynamic model for which
        # the posterior is analytically known? Doing 10_000 samples to run the test suite
        # is not ideal
        # Issue ref: https://github.com/TuringLang/Turing.jl/issues/2402
        @test isapprox(mean(num_ms), 8.6087; atol=0.8)
        @test isapprox(std(num_ms), 1.8865; atol=0.02)
    end

    # The below test used to sample incorrectly before
    # https://github.com/TuringLang/Turing.jl/pull/2328
    @testset "dynamic model with ESS" begin
        @model function dynamic_model_for_ess()
            b ~ Bernoulli()
            x_length = b ? 1 : 2
            x = Vector{Float64}(undef, x_length)
            for i in 1:x_length
                x[i] ~ Normal(i, 1.0)
            end
        end

        m = dynamic_model_for_ess()
        chain = sample(m, Gibbs(:b => PG(10), :x => ESS()), 2000; discard_initial=100)
        means = Dict(:b => 0.5, "x[1]" => 1.0, "x[2]" => 2.0)
        stds = Dict(:b => 0.5, "x[1]" => 1.0, "x[2]" => 1.0)
        for vn in keys(means)
            @test isapprox(mean(skipmissing(chain[:, vn, 1])), means[vn]; atol=0.1)
            @test isapprox(std(skipmissing(chain[:, vn, 1])), stds[vn]; atol=0.1)
        end
    end

    @testset "dynamic model with dot tilde" begin
        @model function dynamic_model_with_dot_tilde(
            num_zs=10, ::Type{M}=Vector{Float64}
        ) where {M}
            z = M(undef, num_zs)
            z .~ Poisson(1.0)
            num_ms = sum(z)
            m = M(undef, num_ms)
            return m .~ Normal(1.0, 1.0)
        end
        model = dynamic_model_with_dot_tilde()
        # TODO(mhauru) This is broken because of
        # https://github.com/TuringLang/DynamicPPL.jl/issues/700.
        @test_broken (
            sample(model, Gibbs(:z => PG(10), :m => HMC(0.01, 4; adtype=adbackend)), 100);
            true
        )
    end

    @testset "Demo models" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            vns = DynamicPPL.TestUtils.varnames(model)
            samplers = [
                Turing.Gibbs(@varname(s) => NUTS(), @varname(m) => NUTS()),
                Turing.Gibbs(@varname(s) => NUTS(), @varname(m) => HMC(0.01, 4)),
                Turing.Gibbs(@varname(s) => NUTS(), @varname(m) => ESS()),
            ]

            if !has_dot_assume(model)
                # Add in some MH samplers, which are not compatible with `.~`.
                append!(
                    samplers,
                    [
                        Turing.Gibbs(@varname(s) => HMC(0.01, 4), @varname(m) => MH()),
                        Turing.Gibbs(@varname(s) => MH(), @varname(m) => HMC(0.01, 4)),
                    ],
                )
            end

            @testset "$sampler" for sampler in samplers
                # Check that taking steps performs as expected.
                rng = Random.default_rng()
                transition, state = AbstractMCMC.step(
                    rng, model, DynamicPPL.Sampler(sampler)
                )
                check_transition_varnames(transition, vns)
                for _ in 1:5
                    transition, state = AbstractMCMC.step(
                        rng, model, DynamicPPL.Sampler(sampler), state
                    )
                    check_transition_varnames(transition, vns)
                end
            end

            # Run the Gibbs sampler and NUTS on the same model, compare statistics of the
            # chains.
            @testset "comparison with 'gold-standard' samples" begin
                num_iterations = 1_000
                thinning = 10
                num_chains = 4

                # Determine initial parameters to make comparison as fair as possible.
                posterior_mean = DynamicPPL.TestUtils.posterior_mean(model)
                initial_params = DynamicPPL.TestUtils.update_values!!(
                    DynamicPPL.VarInfo(model),
                    posterior_mean,
                    DynamicPPL.TestUtils.varnames(model),
                )[:]
                initial_params = fill(initial_params, num_chains)

                # Sampler to use for Gibbs components.
                hmc = HMC(0.1, 32)
                sampler = Turing.Gibbs(@varname(s) => hmc, @varname(m) => hmc)
                Random.seed!(42)
                chain = sample(
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
                Random.seed!(42)
                chain_true = sample(
                    model,
                    NUTS(),
                    MCMCThreads(),
                    num_iterations,
                    num_chains;
                    progress=false,
                    initial_params=initial_params,
                    thinning=thinning,
                )

                # Perform KS test to ensure that the chains are similar.
                xs = Array(chain)
                xs_true = Array(chain_true)
                for i in 1:size(xs, 2)
                    @test two_sample_test(xs[:, i], xs_true[:, i]; warn_on_fail=true)
                    # Let's make sure that the significance level is not too low by
                    # checking that the KS test fails for some simple transformations.
                    # TODO: Replace the heuristic below with closed-form implementations
                    # of the targets, once they are implemented in DynamicPPL.
                    @test !two_sample_test(0.9 .* xs_true[:, i], xs_true[:, i])
                    @test !two_sample_test(1.1 .* xs_true[:, i], xs_true[:, i])
                    @test !two_sample_test(1e-1 .+ xs_true[:, i], xs_true[:, i])
                end
            end
        end
    end

    @testset "multiple varnames" begin
        rng = Random.default_rng()

        @testset "with both `s` and `m` as random" begin
            model = gdemo(1.5, 2.0)
            vns = (@varname(s), @varname(m))
            alg = Turing.Gibbs(vns => MH())

            # `step`
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
            check_transition_varnames(transition, vns)
            for _ in 1:5
                transition, state = AbstractMCMC.step(
                    rng, model, DynamicPPL.Sampler(alg), state
                )
                check_transition_varnames(transition, vns)
            end

            # `sample`
            Random.seed!(42)
            chain = sample(model, alg, 1_000; progress=false)
            check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.4)
        end

        @testset "without `m` as random" begin
            model = gdemo(1.5, 2.0) | (m=7 / 6,)
            vns = (@varname(s),)
            alg = Turing.Gibbs(vns => MH())

            # `step`
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
            check_transition_varnames(transition, vns)
            for _ in 1:5
                transition, state = AbstractMCMC.step(
                    rng, model, DynamicPPL.Sampler(alg), state
                )
                check_transition_varnames(transition, vns)
            end
        end
    end

    @testset "CSMC + ESS" begin
        rng = Random.default_rng()
        model = MoGtest_default
        alg = Turing.Gibbs(
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
        transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
        check_transition_varnames(transition, vns)
        for _ in 1:5
            transition, state = AbstractMCMC.step(
                rng, model, DynamicPPL.Sampler(alg), state
            )
            check_transition_varnames(transition, vns)
        end

        # Sample!
        Random.seed!(42)
        chain = sample(MoGtest_default, alg, 1000; progress=false)
        check_MoGtest_default(chain; atol=0.2)
    end

    @testset "CSMC + ESS (usage of implicit varname)" begin
        rng = Random.default_rng()
        model = MoGtest_default_z_vector
        alg = Turing.Gibbs(
            @varname(z) => CSMC(15), @varname(mu1) => ESS(), @varname(mu2) => ESS()
        )
        vns = (
            @varname(z[1]),
            @varname(z[2]),
            @varname(z[3]),
            @varname(z[4]),
            @varname(mu1),
            @varname(mu2)
        )
        # `step`
        transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
        check_transition_varnames(transition, vns)
        for _ in 1:5
            transition, state = AbstractMCMC.step(
                rng, model, DynamicPPL.Sampler(alg), state
            )
            check_transition_varnames(transition, vns)
        end

        # Sample!
        Random.seed!(42)
        chain = sample(model, alg, 1000; progress=false)
        check_MoGtest_default_z_vector(chain; atol=0.2)
    end

    @testset "externsalsampler" begin
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
            sampler = Turing.Gibbs(
                @varname(m1) => sampler_inner, @varname(m2) => sampler_inner
            )
            Random.seed!(42)
            chain = sample(
                model, sampler, 1000; discard_initial=1000, thinning=10, n_adapts=0
            )
            check_numerical(chain, [:m1, :m2], [-0.2, 0.6]; atol=0.1)
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
end

end
