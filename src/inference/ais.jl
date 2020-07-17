## Sampler

# algorithm
# TODO: actually implement structs for proposal, ratio and schedule - maybe do something similar to mh.jl

struct AIS <: InferenceAlgorithm 
    num_steps :: Integer # should I use some other type here?
    proposal :: P
    ratio :: R 
    schedule :: S
end

# state: same as for vanilla IS

mutable struct AISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                 ::  V
    final_logevidence  ::  F
end

AISState(model::Model) = AISState(VarInfo(model), 0.0)

# Sampler constructor: same as for vanilla IS

function Sampler(alg::AIS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ISState(model)
    return Sampler(alg, info, s, state)
end


## implement abstractMCMC

# TODO: here I chose to do things inside step! like in vanilla IS, but perhaps it would be better to do things inside sample_init! as in SMC?

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    ::Integer,
    transition;
    kwargs...
)
    # TODO: modify - for now this is the same as in IS
    empty!(spl.state.vi)
    model(rng, spl.state.vi, spl)

    return Transition(spl)
end


# sample_end! combines the individual weights to obtain final_logevidence, as in vanilla IS 

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # Calculate evidence.
    spl.state.final_logevidence = logsumexp(map(x->x.lp, ts)) - log(N)
end


## overload assume and observe

# don't know exactly how i'll proceed here...