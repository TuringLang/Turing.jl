module TuringADTests

using Turing
using DynamicPPL
using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using StableRNGs: StableRNG
using Test
using ..Models: gdemo_default
using ..ADUtils: ADTypeCheckContext, adbackends

@testset verbose = true "AD / SamplingContext" begin
    # AD tests for gradient-based samplers need to be run with SamplingContext
    # because samplers can potentially use this to define custom behaviour in
    # the tilde-pipeline and thus change the code executed during model
    # evaluation.
    @testset "adtype=$adtype" for adtype in adbackends
        @testset "alg=$alg" for alg in [
            HMC(0.1, 10; adtype=adtype),
            HMCDA(0.8, 0.75; adtype=adtype),
            NUTS(1000, 0.8; adtype=adtype),
            SGHMC(; learning_rate=0.02, momentum_decay=0.5, adtype=adtype),
            SGLD(; stepsize=PolynomialStepsize(0.25), adtype=adtype),
        ]
            @info "Testing AD for $alg"

            @testset "model=$(model.f)" for model in DEMO_MODELS
                rng = StableRNG(123)
                ctx = DynamicPPL.SamplingContext(rng, DynamicPPL.Sampler(alg))
                @test run_ad(model, adtype; context=ctx, test=true, benchmark=false) isa Any
            end
        end

        @testset "Check ADType" begin
            seed = 123
            alg = HMC(0.1, 10; adtype=adtype)
            m = DynamicPPL.contextualize(
                gdemo_default, ADTypeCheckContext(adtype, gdemo_default.context)
            )
            # These will error if the adbackend being used is not the one set.
            sample(StableRNG(seed), m, alg, 10)
        end
    end
end

@testset verbose = true "AD / GibbsContext" begin
    # Gibbs sampling also needs extra AD testing because the models are
    # executed with GibbsContext and a subsetted varinfo. (see e.g.
    # `gibbs_initialstep_recursive` and `gibbs_step_recursive` in
    # src/mcmc/gibbs.jl -- the code here mimics what happens in those
    # functions)
    @testset "adtype=$adtype" for adtype in adbackends
        @testset "model=$(model.f)" for model in DEMO_MODELS
            # All the demo models have variables `s` and `m`, so we'll pretend
            # that we're using a Gibbs sampler where both of them are sampled
            # with a gradient-based sampler (say HMC(0.1, 10)).
            # This means we need to construct one with only `s`, and one model with
            # only `m`.
            global_vi = DynamicPPL.VarInfo(model)
            @testset for varnames in ([@varname(s)], [@varname(m)])
                @info "Testing Gibbs AD with model=$(model.f), varnames=$varnames"
                conditioned_model = Turing.Inference.make_conditional(
                    model, varnames, deepcopy(global_vi)
                )
                rng = StableRNG(123)
                ctx = DynamicPPL.SamplingContext(rng, DynamicPPL.Sampler(HMC(0.1, 10)))
                @test run_ad(model, adtype; context=ctx, test=true, benchmark=false) isa Any
            end
        end
    end
end

end # module
