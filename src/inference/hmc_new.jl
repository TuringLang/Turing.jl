struct TuringState{S,F}
    state::S
    logdensity::F
end

state_to_turing(f::DynamicPPL.LogDensityFunction, state) = TuringState(state, f)
function transition_to_turing(f::DynamicPPL.LogDensityFunction, transition)
    θ = getparams(transition)
    varinfo = DynamicPPL.unflatten(f.varinfo, θ)
    # TODO: `deepcopy` is overkill; make more efficient.
    varinfo = DynamicPPL.invlink!!(deepcopy(varinfo), f.model)
    return HMCTransition(varinfo, transition)
end

# NOTE: Can easily be implemented for other samplers.
getparams(transition::AdvancedHMC.Transition) = transition.z.θ


function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AdvancedHMC.HMCSampler;
    kwargs...
)
    # Create a log-density function.
    f = DynamicPPL.LogDensityFunction(model)

    # Link the varinfo.
    DynamicPPL.Setfield.@set! f.varinfo = link!!(f.varinfo, model)

    # Then just call `AdvancedHMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(f), sampler; kwargs...)

    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AdvancedHMC.HMCSampler,
    state::TuringState;
    kwargs...
)
    f = state.logdensity

    # Then just call `AdvancedHMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...)

    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end

# TODO: This needs to be replaced by something much better.
# We can probably re-use a lot from `DynamicPPL.initialstep`.
function initialize_nuts(model::Turing.Model)
    f = DynamicPPL.LogDensityFunction(model)
    DynamicPPL.Setfield.@set! f.varinfo = DynamicPPL.link!!(f.varinfo, f.model)
    
    # Choose parameter dimensionality and initial parameter value
    D = LogDensityProblems.dimension(f)
    initial_θ = rand(D)

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
    proposal = AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS,AdvancedHMC.GeneralisedNoUTurn}(integrator)
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(0.8, integrator)
    )

    return AdvancedHMC.HMCSampler(proposal, metric, adaptor)
end
