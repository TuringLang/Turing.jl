# TODO: decide how to handle multiple particles. a few possibilities:
# 1. each time we call step!, we create a new particle as a transition (like in is.jl, problem = lack of parallelization)
# 2.. this whole file focuses on a single particle, parallelization is handled by sample at a higher level (problem: where to combine particles to estimate log-evidence?)
# 3. something more like in advancedPS (more complex, doesn't seem necessary?)

## Sampler

# algorithm
# TODO: actually implement structs for proposal, ratio and schedule


## TODO: check out how RandomWalkProposal and MH handle these
struct Proposal 
end

@enum Ratio Barker Metropolis

struct AIS <: InferenceAlgorithm 
    num_steps :: Integer # number of MCMC steps = number of intermediate distributions + 1
    proposal :: Proposal
    ratio :: Ratio # can be Barker or Metropolis - check out enums
    schedule :: S # probably a list - that must contain num_steps elements
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

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.vi)
    for current_target in 1:num_steps
        # TODO: define new target using schedule
        # TODO: perform MCMC transition
        # TODO: update weight
    end

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