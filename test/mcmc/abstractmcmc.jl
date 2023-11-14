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


function initialize_mh(model)
    f = DynamicPPL.LogDensityFunction(model)
    d = LogDensityProblems.dimension(f)
    return AdvancedMH.RWMH(MvNormal(Zeros(d), 0.1 * I))
end

@testset "External samplers" begin
    @turing_testset "AdvancedHMC.jl" begin
        for model in DynamicPPL.TestUtils.DEMO_MODELS
            # Need some functionality to initialize the sampler.
            # TODO: Remove this once the constructors in the respective packages become "lazy".
            sampler = initialize_nuts(model);
            DynamicPPL.TestUtils.test_sampler(
                [model],
                DynamicPPL.Sampler(externalsampler(sampler), model),
                5_000;
                n_adapts=1_000,
                discard_initial=1_000,
                rtol=0.2,
                sampler_name="AdvancedHMC"
            )
        end
    end

    @turing_testset "AdvancedMH.jl" begin
        for model in DynamicPPL.TestUtils.DEMO_MODELS
            # Need some functionality to initialize the sampler.
            # TODO: Remove this once the constructors in the respective packages become "lazy".
            sampler = initialize_mh(model);
            DynamicPPL.TestUtils.test_sampler(
                [model],
                DynamicPPL.Sampler(externalsampler(sampler), model),
                10_000;
                discard_initial=1_000,
                thinning=10,
                rtol=0.2,
                sampler_name="AdvancedMH"
            )
        end
    end
end
