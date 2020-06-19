"""
doesn't work yet
more or less copy pasted IS which is quite similar to ABC
"""

# distance between data
# maybe build an abstract struct distance and build implementations
struct Distance end

# if Distance is an abstract struct, write one method per implementation
struct ABC{space, F<:AbstractFloat} <: InferenceAlgorithm 
    distance :: Distance # metric to measure difference between generated samples and true data
    ϵ :: F # tolerance: how small the distance has to be to accept sampled parameter
end

ABC() = ABC{()}()

SamplerState(model::Model) = ABCState(VarInfo(model), 0.0)

function Sampler(alg::ABC, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = SamplerState(model)
    return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ABC},
    ::Integer,
    transition;
    kwargs...
)   
    dist = 0.
    while dist < ϵ
        empty!(spl.state.vi) # the previous parameter lead to too big a distance -> discard from sampler
        model(rng, spl.state.vi, spl) # TODO: calls to this must add the model parameters to vi and return the value of the observed variables
        # TODO: access the actual data from the model
        # TODO: compute distance between actual data and simulated data

    return Transition(spl)
end

function DynamicPPL.assume(rng, spl::Sampler{<:ABC}, dist::Distribution, vn::VarName, vi)
    # TODO: i think this is what i want to do
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:ABC}, dist::Distribution, value, vi)
    # TODO: unlike for IS, I actually want to sample from this!
    # acclogp!(vi, logpdf(dist, value))
    return logpdf(dist, value)
end