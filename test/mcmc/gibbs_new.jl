using Turing, DynamicPPL

# Okay, so what do we actually need to test here.
# 1. Needs to be compatible with most models.
# 2. Restricted to usage of pairs for now to make things simple.

# TODO: Don't require usage of tuples due to potential of blowing up compilation times.

# FIXME: Currently failing for `demo_assume_index_observe`.
# Likely an issue with not linking correctly.
@testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
    vns = DynamicPPL.TestUtils.varnames(model)
    # Run one sampler on variables starting with `s` and another on variables starting with `m`.
    vns_s = filter(vns) do vn
        DynamicPPL.getsym(vn) == :s
    end
    vns_m = filter(vns) do vn
        DynamicPPL.getsym(vn) == :m
    end

    # Construct the sampler.
    sampler = Turing.Inference.GibbsV2(
        vns_s => Turing.Inference.NUTS(),
        vns_m => Turing.Inference.NUTS(),
    )

    # Check that taking steps performs as expected.
    rng = Random.default_rng()
    transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler))
    @test keys(transition.θ) == Tuple(unique(map(DynamicPPL.getsym, vns)))

    for _ = 1:10
        transition, state =
            AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler), state)
        @test keys(transition.θ) == Tuple(unique(map(DynamicPPL.getsym, vns)))
    end
end

# @testset "Gibbs using `condition`" begin
#     @testset "demo_assume_dot_observe" begin
#         model = DynamicPPL.TestUtils.demo_assume_dot_observe()
#         # Construct the different varinfos to be used.
#         varinfos = (SimpleVarInfo(s = 1.0), SimpleVarInfo(m = 10.0))
#         # Construct the varinfo for the particular variable we want to sample.
#         target_varinfo = first(varinfos)

#         # Create the conditional model.
#         conditional_model =
#             Turing.Inference.make_conditional(model, target_varinfo, varinfos)

#         # Sample!
#         sampler = Turing.Inference.GibbsV2(@varname(s) => MH(), @varname(m) => MH())
#         rng = Random.default_rng()

#         @testset "step" begin
#             transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler))
#             @test keys(transition.θ) == (:s, :m)

#             transition, state =
#                 AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler), state)
#             @test keys(transition.θ) == (:s, :m)

#             transition, state =
#                 AbstractMCMC.step(rng, model, DynamicPPL.Sampler(sampler), state)
#             @test keys(transition.θ) == (:s, :m)
#         end

#         @testset "sample" begin
#             chain = sample(model, sampler, 1000)
#             @test size(chain, 1) == 1000
#             display(mean(chain))
#         end
#     end

#     # @testset "gdemo" begin
#     #     Random.seed!(100)
#     #     alg = Turing.Inference.GibbsV2(@varname(s) => CSMC(15), @varname(m) => ESS(:m))
#     #     chain = sample(gdemo(1.5, 2.0), alg, 10_000)
#     # end

#     # @testset "MoGtest" begin
#     #     Random.seed!(125)
#     #     alg = Gibbs(CSMC(15, :z1, :z2, :z3, :z4), ESS(:mu1), ESS(:mu2))
#     #     chain = sample(MoGtest_default, alg, 6000)
#     #     check_MoGtest_default(chain, atol = 0.1)
#     # end

#     @testset "multiple varnames" begin
#         rng = Random.default_rng()

#         # With both `s` and `m` as random.
#         model = gdemo(1.5, 2.0)
#         alg = Turing.Inference.GibbsV2((@varname(s), @varname(m)) => MH())

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
#         @test keys(transition.θ) == (:s, :m)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:s, :m)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:s, :m)

#         # Sample.
#         chain = sample(model, alg, 10_000)
#         check_numerical(chain, [:s, :m], [49 / 24, 7 / 6], atol = 0.1)


#         # Without `m` as random.
#         model = gdemo(1.5, 2.0) | (m = 7 / 6,)
#         alg = Turing.Inference.GibbsV2((@varname(s),) => MH())
#         @info "" alg alg.varnames

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
#         @test keys(transition.θ) == (:s,)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:s,)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:s,)
#     end

#     @testset "CSMS + ESS" begin
#         rng = Random.default_rng()
#         model = MoGtest_default
#         alg = Turing.Inference.GibbsV2(
#             (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
#             @varname(mu1) => ESS(),
#             @varname(mu2) => ESS(),
#         )
#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg))
#         @test keys(transition.θ) == (:mu1, :mu2, :z1, :z2, :z3, :z4)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:mu1, :mu2, :z1, :z2, :z3, :z4)

#         transition, state = AbstractMCMC.step(rng, model, DynamicPPL.Sampler(alg), state)
#         @test keys(transition.θ) == (:mu1, :mu2, :z1, :z2, :z3, :z4)

#         # Sample!
#         chain = sample(MoGtest_default, alg, 1000)
#         check_MoGtest_default(chain, atol = 0.1)
#     end
# end
