using Turing.Inference: AdvancedHMC

function initialize_nuts(model::Turing.Model)
    # Create a log-density function with an implementation of the
    # gradient so we ensure that we're using the same AD backend as in Turing.
    f = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(model))

    # Link the varinfo.
    f = Turing.Inference.setvarinfo(f, DynamicPPL.link!!(Turing.Inference.getvarinfo(f), model))

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
    proposal = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(0.65, integrator)
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
struct ModelDistribution{M<:DynamicPPL.Model,V<:DynamicPPL.VarInfo} <: ContinuousMultivariateDistribution
    model::M
    varinfo::V
end
ModelDistribution(model::DynamicPPL.Model) = ModelDistribution(model, DynamicPPL.VarInfo(model))

Base.length(d::ModelDistribution) = length(d.varinfo[:])
function Distributions._logpdf(d::ModelDistribution, x::AbstractVector)
    return logprior(d.model, DynamicPPL.unflatten(d.varinfo, x))
end
function Distributions._rand!(rng::Random.AbstractRNG, d::ModelDistribution, x::AbstractVector{<:Real})
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
    f = DynamicPPL.LogDensityFunction(model)
    d = LogDensityProblems.dimension(f)
    return AdvancedMH.MetropolisHastings(AdvancedMH.StaticProposal(ModelDistribution(model)))
end

@testset "External samplers" begin
    @turing_testset "AdvancedHMC.jl" begin
        # Try a few different AD backends.
        @testset "adtype=$adtype" for adtype in [AutoForwardDiff(), AutoReverseDiff()]
            for model in DynamicPPL.TestUtils.DEMO_MODELS
                # Need some functionality to initialize the sampler.
                # TODO: Remove this once the constructors in the respective packages become "lazy".
                sampler = initialize_nuts(model);
                DynamicPPL.TestUtils.test_sampler(
                    [model],
                    DynamicPPL.Sampler(externalsampler(sampler; adtype, unconstrained=true), model),
                    5_000;
                    n_adapts=1_000,
                    discard_initial=1_000,
                    rtol=0.2,
                    sampler_name="AdvancedHMC"
                )
            end
        end
    end

    @turing_testset "AdvancedMH.jl" begin
        @testset "RWMH" begin
            for model in DynamicPPL.TestUtils.DEMO_MODELS
                # Need some functionality to initialize the sampler.
                # TODO: Remove this once the constructors in the respective packages become "lazy".
                sampler = initialize_mh_rw(model);
                DynamicPPL.TestUtils.test_sampler(
                    [model],
                    DynamicPPL.Sampler(externalsampler(sampler; unconstrained=true), model),
                    10_000;
                    discard_initial=1_000,
                    thinning=10,
                    rtol=0.2,
                    sampler_name="AdvancedMH"
                )
            end
        end
        @testset "MH with prior proposal" begin
            for model in DynamicPPL.TestUtils.DEMO_MODELS
                sampler = initialize_mh_with_prior_proposal(model);
                DynamicPPL.TestUtils.test_sampler(
                    [model],
                    DynamicPPL.Sampler(externalsampler(sampler; unconstrained=false), model),
                    10_000;
                    discard_initial=1_000,
                    rtol=0.2,
                    sampler_name="AdvancedMH"
                )
            end
        end
    end
end
