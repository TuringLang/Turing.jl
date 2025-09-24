module ExternalSamplerTests

using ..Models: gdemo_default
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

@testset "External sampler interface" begin
    # Turing declares an interface for external samplers (see docstring for
    # ExternalSampler). We should check that implementing this interface
    # and only this interface allows us to use the sampler in Turing.
    struct MyTransition{V<:AbstractVector}
        params::V
    end
    # Samplers need to implement `Turing.Inference.getparams`.
    Turing.Inference.getparams(::DynamicPPL.Model, t::MyTransition) = t.params
    # State doesn't matter (but we need to carry the params through to the next
    # iteration).
    struct MyState{V<:AbstractVector}
        params::V
    end

    # externalsamplers must accept LogDensityModel inside their step function.
    # By default Turing gives the externalsampler a LDF constructed with
    # adtype=ForwardDiff, so we should expect that inside the sampler we can
    # call both `logdensity` and `logdensity_and_gradient`.
    #
    # The behaviour of this sampler is to simply calculate logp and its
    # gradient, and then return the same values.
    #
    # TODO: Do we also want to run ADTypeCheckContext to make sure that it is 
    # indeed using the adtype provided from Turing?
    struct MySampler <: AbstractMCMC.AbstractSampler end
    function AbstractMCMC.step(
        rng::Random.AbstractRNG,
        model::AbstractMCMC.LogDensityModel,
        sampler::MySampler;
        initial_params::AbstractVector,
        kwargs...,
    )
        # Step 1
        ldf = model.logdensity
        lp = LogDensityProblems.logdensity(ldf, initial_params)
        @test lp isa Real
        lp, grad = LogDensityProblems.logdensity_and_gradient(ldf, initial_params)
        @test lp isa Real
        @test grad isa AbstractVector{<:Real}
        return MyTransition(initial_params), MyState(initial_params)
    end
    function AbstractMCMC.step(
        rng::Random.AbstractRNG,
        model::AbstractMCMC.LogDensityModel,
        sampler::MySampler,
        state::MyState;
        kwargs...,
    )
        # Step >= 1
        params = state.params
        ldf = model.logdensity
        lp = LogDensityProblems.logdensity(ldf, params)
        @test lp isa Real
        lp, grad = LogDensityProblems.logdensity_and_gradient(ldf, params)
        @test lp isa Real
        @test grad isa AbstractVector{<:Real}
        return MyTransition(params), MyState(params)
    end

    @model function test_external_sampler()
        a ~ Beta(2, 2)
        return b ~ Normal(a)
    end
    model = test_external_sampler()
    a, b = 0.5, 0.0

    chn = sample(model, externalsampler(MySampler()), 10; initial_params=[a, b])
    @test chn isa MCMCChains.Chains
    @test all(chn[:a] .== a)
    @test all(chn[:b] .== b)
    expected_logpdf = logpdf(Beta(2, 2), a) + logpdf(Normal(a), b)
    @test all(chn[:lp] .== expected_logpdf)
    @test all(chn[:logprior] .== expected_logpdf)
    @test all(chn[:loglikelihood] .== 0.0)
end

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
    _, varinfo = DynamicPPL.init!!(rng, model, varinfo, DynamicPPL.InitFromPrior())
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

@testset verbose = true "Implementation of externalsampler interface for known packages" begin
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
