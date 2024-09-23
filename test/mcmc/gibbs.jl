module GibbsTests

using ..Models: MoGtest_default, MoGtest_default_z_vector, gdemo, gdemo_default
using ..NumericalTests:
    check_MoGtest_default,
    check_MoGtest_default_z_vector,
    check_gdemo,
    check_numerical,
    two_sample_test
import ..ADUtils
using Distributions: InverseGamma, Normal
using Distributions: sample
using DynamicPPL: DynamicPPL
using ForwardDiff: ForwardDiff
using Random: Random
using ReverseDiff: ReverseDiff
using Test: @test, @testset
using Turing
using Turing: Inference
using Turing.Inference: AdvancedHMC, AdvancedMH
using Turing.RandomMeasures: ChineseRestaurantProcess, DirichletProcess

ADUtils.install_tapir && import Tapir

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
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_observe_literal)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_literal_dot_observe)},
    DynamicPPL.Model{typeof(DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix)},
}
has_dot_assume(::DEMO_MODELS_WITHOUT_DOT_ASSUME) = false
has_dot_assume(::DynamicPPL.Model) = true

@testset "Testing gibbs.jl with $adbackend" for adbackend in ADUtils.adbackends
    @testset "gibbs constructor" begin
        N = 500
        s1 = begin
            alg = HMC(0.1, 5, :s, :m; adtype=adbackend)
            Gibbs(; s=alg, m=alg)
        end
        s2 = begin
            alg = PG(10)
            Gibbs(@varname(s) => alg, @varname(m) => alg)
        end
        s3 = Gibbs((; s=PG(3), m=HMC(0.4, 8; adtype=adbackend)))
        s4 = Gibbs(Dict(@varname(s) => PG(3), @varname(m) => HMC(0.4, 8; adtype=adbackend)))
        s5 = Gibbs(; s=CSMC(3), m=HMC(0.4, 8; adtype=adbackend))
        s6 = Gibbs(; s=HMC(0.1, 5; adtype=adbackend), m=ESS())
        s7 = Gibbs((@varname(s), @varname(m)) => PG(10))
        for s in (s1, s2, s3, s4, s5, s6, s7)
            @test DynamicPPL.alg_str(Turing.Sampler(s, gdemo_default)) == "Gibbs"
        end

        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
        c5 = sample(gdemo_default, s5, N)
        c6 = sample(gdemo_default, s6, N)
        c7 = sample(gdemo_default, s7, N)

        g = Turing.Sampler(s3, gdemo_default)
        @test sample(gdemo_default, g, N) isa MCMCChains.Chains
    end

    @testset "gibbs inference" begin
        Random.seed!(100)
        alg = Gibbs(; s=CSMC(15), m=HMC(0.2, 4; adtype=adbackend))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:m], [7 / 6]; atol=0.15)
        # Be more relaxed with the tolerance of the variance.
        check_numerical(chain, [:s], [49 / 24]; atol=0.35)

        Random.seed!(100)

        alg = Gibbs(; s=MH(), m=HMC(0.2, 4; adtype=adbackend))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)

        alg = Gibbs(; s=CSMC(15), m=ESS())
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)

        alg = CSMC(15)
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6]; atol=0.1)

        Random.seed!(200)
        gibbs = Gibbs(
            (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => PG(15),
            (@varname(mu1), @varname(mu2)) => HMC(0.15, 3; adtype=adbackend),
        )
        chain = sample(MoGtest_default, gibbs, 10_000)
        check_MoGtest_default(chain; atol=0.15)

        Random.seed!(200)
        for alg in [
            # The new syntax for specifying a sampler to run twice for one variable.
            Gibbs(s => MH(), s => MH(), m => HMC(0.2, 4; adtype=adbackend)),
            Gibbs(s => MH(), m => HMC(0.2, 4), m => HMC(0.2, 4); adtype=adbackend),
        ]
            chain = sample(gdemo(1.5, 2.0), alg, 10_000)
            check_gdemo(chain; atol=0.15)
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

        alg = Gibbs(; s=MH(), m=HMC(0.2, 4; adtype=adbackend))
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
        model = imm(Random.randn(100), 1.0)
        # https://github.com/TuringLang/Turing.jl/issues/1725
        # sample(model, Gibbs(MH(:z), HMC(0.01, 4, :m)), 100);
        sample(model, Gibbs(; z=PG(10), m=HMC(0.01, 4; adtype=adbackend)), 100)
    end

    @testset "Demo models" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            vns = DynamicPPL.TestUtils.varnames(model)
            # Run one sampler on variables starting with `s` and another on variables starting with `m`.
            vns_s = filter(vns) do vn
                DynamicPPL.getsym(vn) == :s
            end
            vns_m = filter(vns) do vn
                DynamicPPL.getsym(vn) == :m
            end

            samplers = [
                Turing.Gibbs(vns_s => NUTS(), vns_m => NUTS()),
                Turing.Gibbs(vns_s => NUTS(), vns_m => HMC(0.01, 4)),
            ]

            if !has_dot_assume(model)
                # Add in some MH samplers, which are not compatible with `.~`.
                append!(
                    samplers,
                    [
                        Turing.Gibbs(vns_s => HMC(0.01, 4), vns_m => MH()),
                        Turing.Gibbs(vns_s => MH(), vns_m => HMC(0.01, 4)),
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
                sampler_inner = HMC(0.1, 32)
                sampler = Turing.Gibbs(vns_s => sampler_inner, vns_m => sampler_inner)
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
            chain = sample(model, alg, 10_000; progress=false)
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
end

end
