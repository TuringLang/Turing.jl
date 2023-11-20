using Test, Random, Turing, DynamicPPL

function check_transition_varnames(
    transition::Turing.Inference.Transition,
    parent_varnames
)
    transition_varnames = mapreduce(vcat, transition.θ) do vn_and_val
        [first(vn_and_val)]
    end
    # Varnames in `transition` should be subsumed by those in `vns`.
    for vn in transition_varnames
        @test any(Base.Fix2(DynamicPPL.subsumes, vn), parent_varnames)
    end
end

# Okay, so what do we actually need to test here.
# 1. (✓) Needs to be compatible with most models.
# 2. (???) Restricted to usage of pairs for now to make things simple.

# TODO: Don't require usage of tuples due to potential of blowing up compilation times.

const DEMO_MODELS_WITHOUT_DOT_ASSUME = Union{
    Model{typeof(DynamicPPL.TestUtils.demo_assume_index_observe)},
    Model{typeof(DynamicPPL.TestUtils.demo_assume_multivariate_observe)},
    Model{typeof(DynamicPPL.TestUtils.demo_assume_dot_observe)},
    Model{typeof(DynamicPPL.TestUtils.demo_assume_observe_literal)},
    Model{typeof(DynamicPPL.TestUtils.demo_assume_literal_dot_observe)},
    Model{typeof(DynamicPPL.TestUtils.demo_assume_matrix_dot_observe_matrix)},
}
has_dot_assume(::DEMO_MODELS_WITHOUT_DOT_ASSUME) = false
has_dot_assume(::Model) = true

# Likely an issue with not linking correctly.
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
            GibbsV2(
                vns_s => NUTS(),
                vns_m => NUTS(),
            ),
            GibbsV2(
                vns_s => NUTS(),
                vns_m => HMC(0.01, 4),
            )
        ]

        if !has_dot_assume(model)
            # Add in some MH samplers, which are not compatible with `.~`.
            append!(
                samplers,
                [
                    GibbsV2(
                        vns_s => HMC(0.01, 4),
                        vns_m => MH(),
                    ),
                    GibbsV2(
                        vns_s => MH(),
                        vns_m => HMC(0.01, 4),
                    )
                ]
            )
        end

        @testset "$sampler" for sampler in samplers
            # Check that taking steps performs as expected.
            rng = Random.default_rng()
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler))
            check_transition_varnames(transition, vns)
            for _ = 1:5
                transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler), state)
                check_transition_varnames(transition, vns)
            end
        end
    end
end

@testset "Gibbs using `condition`" begin
    @testset "demo_assume_dot_observe" begin
        model = DynamicPPL.TestUtils.demo_assume_dot_observe()

        # Sample!
        rng = Random.default_rng()
        vns = [@varname(s), @varname(m)]
        sampler = GibbsV2(map(Base.Fix2(Pair, MH()), vns)...)

        @testset "step" begin
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler))
            check_transition_varnames(transition, vns)
            for _ = 1:5
                transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler), state)
                check_transition_varnames(transition, vns)
            end
        end

        @testset "sample" begin
            chain = sample(model, sampler, 1000; progress=false)
            @test size(chain, 1) == 1000
            display(mean(chain))
        end
    end

    @testset "gdemo with CSMC & ESS" begin
        # `GibbsV2` does not work with SMC samplers, e.g. `CSMC`.
        # FIXME: Oooor it is (see tests below). Uncertain.
        Random.seed!(100)
        alg = GibbsV2(@varname(s) => CSMC(15), @varname(m) => ESS(:m))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_gdemo(chain)
    end

    @testset "multiple varnames" begin
        rng = Random.default_rng()

        # With both `s` and `m` as random.
        model = gdemo(1.5, 2.0)
        vns = (@varname(s), @varname(m))
        alg = GibbsV2(vns => MH())

        # `step`
        transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
        check_transition_varnames(transition, vns)
        for _ = 1:5
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
            check_transition_varnames(transition, vns)
        end

        # `sample`
        chain = sample(model, alg, 10_000; progress=false)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol = 0.1)

        # Without `m` as random.
        model = gdemo(1.5, 2.0) | (m = 7 / 6,)
        vns = (@varname(s),)
        alg = GibbsV2(vns => MH())

        # `step`
        transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
        check_transition_varnames(transition, vns)
        for _ = 1:5
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
            check_transition_varnames(transition, vns)
        end
    end

    @testset "CSMC + ESS" begin
        rng = Random.default_rng()
        model = MoGtest_default
        alg = GibbsV2(
            (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
            @varname(mu1) => ESS(),
            @varname(mu2) => ESS(),
        )
        vns = (@varname(z1), @varname(z2), @varname(z3), @varname(z4), @varname(mu1), @varname(mu2))
        # `step`
        transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
        check_transition_varnames(transition, vns)
        for _ = 1:5
            transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
            check_transition_varnames(transition, vns)
        end

        # Sample!
        chain = sample(MoGtest_default, alg, 1000; progress=true)
        check_MoGtest_default(chain, atol = 0.2)
    end
end
