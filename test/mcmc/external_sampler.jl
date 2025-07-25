module ExternalSamplerTests

using AbstractMCMC: AbstractMCMC
using AdvancedMH: AdvancedMH
using Distributions: sample
using Distributions.FillArrays: Zeros
using DynamicPPL: DynamicPPL
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems
using Random: Random
using ReverseDiff: ReverseDiff
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
using Turing
using Turing.Inference: AdvancedHMC

function initialize_nuts(model::DynamicPPL.Model)
    # Create a linked varinfo
    vi = DynamicPPL.VarInfo(model)
    linked_vi = DynamicPPL.link!!(vi, model)

    # Create a LogDensityFunction
    f = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, linked_vi; adtype=Turing.DEFAULT_ADTYPE
    )

    # Choose parameter dimensionality and initial parameter value
    D = LogDensityProblems.dimension(f)
    initial_θ = rand(D) .- 0.5

    # Define a Hamiltonian system
    metric = AdvancedHMC.DiagEuclideanMetric(D)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, f)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = AdvancedHMC.find_good_stepsize(hamiltonian, initial_θ)
    integrator = AdvancedHMC.Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator, AdvancedHMC.GeneralisedNoUTurn()
        ),
    )
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.65, integrator)
    )

    return AdvancedHMC.HMCSampler(proposal, metric, adaptor)
end

function initialize_mh_rw(model)
    f = DynamicPPL.LogDensityFunction(model)
    d = LogDensityProblems.dimension(f)
    return AdvancedMH.RWMH(MvNormal(Zeros(d), 0.1 * I))
end

# TODO: Should this go somewhere else?
# Convert a model into a `Distribution` to allow usage as a proposal in AdvancedMH.jl.
struct ModelDistribution{M<:DynamicPPL.Model,V<:DynamicPPL.VarInfo} <:
       ContinuousMultivariateDistribution
    model::M
    varinfo::V
end
function ModelDistribution(model::DynamicPPL.Model)
    return ModelDistribution(model, DynamicPPL.VarInfo(model))
end

Base.length(d::ModelDistribution) = length(d.varinfo[:])
function Distributions._logpdf(d::ModelDistribution, x::AbstractVector)
    return logprior(d.model, DynamicPPL.unflatten(d.varinfo, x))
end
function Distributions._rand!(
    rng::Random.AbstractRNG, d::ModelDistribution, x::AbstractVector{<:Real}
)
    model = d.model
    varinfo = deepcopy(d.varinfo)
    for vn in keys(varinfo)
        DynamicPPL.set_flag!(varinfo, vn, "del")
    end
    DynamicPPL.evaluate!!(model, varinfo, DynamicPPL.SamplingContext(rng))
    x .= varinfo[:]
    return x
end

function initialize_mh_with_prior_proposal(model)
    return AdvancedMH.MetropolisHastings(
        AdvancedMH.StaticProposal(ModelDistribution(model))
    )
end

function test_initial_params(
    model, sampler, initial_params=DynamicPPL.VarInfo(model)[:]; kwargs...
)
    # Execute the transition with two different RNGs and check that the resulting
    # parameter values are the same.
    rng1 = Random.MersenneTwister(42)
    rng2 = Random.MersenneTwister(43)

    transition1, _ = AbstractMCMC.step(rng1, model, sampler; initial_params, kwargs...)
    transition2, _ = AbstractMCMC.step(rng2, model, sampler; initial_params, kwargs...)
    vn_to_val1 = DynamicPPL.OrderedDict(transition1.θ)
    vn_to_val2 = DynamicPPL.OrderedDict(transition2.θ)
    for vn in union(keys(vn_to_val1), keys(vn_to_val2))
        @test vn_to_val1[vn] ≈ vn_to_val2[vn]
    end
end

@testset verbose = true "External samplers" begin
    @testset "AdvancedHMC.jl" begin
        @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            adtype = Turing.DEFAULT_ADTYPE

            # Need some functionality to initialize the sampler.
            # TODO: Remove this once the constructors in the respective packages become "lazy".
            sampler = initialize_nuts(model)
            sampler_ext = DynamicPPL.Sampler(
                externalsampler(sampler; adtype, unconstrained=true)
            )
            # FIXME: Once https://github.com/TuringLang/AdvancedHMC.jl/pull/366 goes through, uncomment.
            # @testset "initial_params" begin
            #     test_initial_params(model, sampler_ext; n_adapts=0)
            # end

            sample_kwargs = (
                n_adapts=1_000,
                discard_initial=1_000,
                # FIXME: Remove this once we can run `test_initial_params` above.
                initial_params=DynamicPPL.VarInfo(model)[:],
            )

            @testset "inference" begin
                DynamicPPL.TestUtils.test_sampler(
                    [model],
                    sampler_ext,
                    2_000;
                    rtol=0.2,
                    sampler_name="AdvancedHMC",
                    sample_kwargs...,
                )
            end
        end

        @testset "logp is set correctly" begin
            @model logp_check() = x ~ Normal()
            model = logp_check()
            sampler = initialize_nuts(model)
            sampler_ext = externalsampler(
                sampler; adtype=Turing.DEFAULT_ADTYPE, unconstrained=true
            )
            chn = sample(logp_check(), Gibbs(@varname(x) => sampler_ext), 100)
            @test isapprox(logpdf.(Normal(), chn[:x]), chn[:lp])
        end
    end

    @testset "AdvancedMH.jl" begin
        @testset "RWMH" begin
            @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
                # Need some functionality to initialize the sampler.
                # TODO: Remove this once the constructors in the respective packages become "lazy".
                sampler = initialize_mh_rw(model)
                sampler_ext = DynamicPPL.Sampler(
                    externalsampler(sampler; unconstrained=true)
                )
                @testset "initial_params" begin
                    test_initial_params(model, sampler_ext)
                end
                @testset "inference" begin
                    DynamicPPL.TestUtils.test_sampler(
                        [model],
                        sampler_ext,
                        2_000;
                        discard_initial=1_000,
                        thinning=10,
                        rtol=0.2,
                        sampler_name="AdvancedMH",
                    )
                end
            end

            @testset "logp is set correctly" begin
                @model logp_check() = x ~ Normal()
                model = logp_check()
                sampler = initialize_mh_rw(model)
                sampler_ext = externalsampler(sampler; unconstrained=true)
                chn = sample(logp_check(), Gibbs(@varname(x) => sampler_ext), 100)
                @test isapprox(logpdf.(Normal(), chn[:x]), chn[:lp])
            end
        end

        # NOTE: Broken because MH doesn't really follow the `logdensity` interface, but calls
        # it with `NamedTuple` instead of `AbstractVector`.
        # @testset "MH with prior proposal" begin
        #     @testset "$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
        #         sampler = initialize_mh_with_prior_proposal(model);
        #         sampler_ext = DynamicPPL.Sampler(externalsampler(sampler; unconstrained=false))
        #         @testset "initial_params" begin
        #             test_initial_params(model, sampler_ext)
        #         end
        #         @testset "inference" begin
        #             DynamicPPL.TestUtils.test_sampler(
        #                 [model],
        #                 sampler_ext,
        #                 10_000;
        #                 discard_initial=1_000,
        #                 rtol=0.2,
        #                 sampler_name="AdvancedMH"
        #             )
        #         end
        #     end
        # end
    end
end

end
