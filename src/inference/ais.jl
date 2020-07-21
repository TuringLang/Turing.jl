# TODO: decide how to handle multiple particles. a few possibilities:
# 1. prefered: each time we call step!, we create a new particle as a transition (like in is.jl - how does that handle parallelization)
# 2. this whole file focuses on a single particle, parallelization handled at a higher level (problem: combining particles to estimate log-evidence?)
# 3. something more like in advancedPS (more complex, doesn't seem necessary?)

## A. Sampler

# A.1. algorithm

# simple version of AIS (not fully general): 
# - sequence of distributions defined by tempering
# - transition kernels are MCMC kernels with the same proposal
# - acceptance ratios vary to ensure invariance

# to simplify even more, maybe require symmetric proposals: no need to evaluate proposal density?

# TODO: check out how RandomWalkProposal and MH handle these
struct Proposal 
end

# TODO: maybe do something for schedule here, maybe not

struct AIS <: InferenceAlgorithm 
    num_steps :: Integer # number of MCMC steps = number of intermediate distributions + 1
    proposal :: Distributions.Sampleable
    schedule :: S # probably a list - that must contain num_steps elements
end

# A.2. state: same as for vanilla IS

mutable struct AISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                 ::  V # reset for every step ie particle
    final_logevidence  ::  F
end

AISState(model::Model) = AISState(VarInfo(model), 0.0)

# A.3. Sampler constructor: same as for vanilla IS

function Sampler(alg::AIS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = AISState(model)
    return Sampler(alg, info, s, state)
end


## B. Implement AbstractMCMC

# B.1. new transition type, with an additional attribute weight

struct Transition{T, F<:AbstractFloat}
    θ  :: T
    lp :: F
    weight :: F
end

function Transition(spl::Sampler, weight::F<:AbstractFloat, nt::NamedTuple=NamedTuple())
    theta = merge(tonamedtuple(spl.state.vi), nt)
    lp = getlogp(spl.state.vi)
    return Transition{typeof(theta), typeof(lp)}(theta, lp, weight)
end

# idk what this function is for
function additional_parameters(::Type{<:Transition})
    return [:lp, :weight]
end

# B.2. step function 

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.vi) # particles are independent: previous step doesn't matter
    # TODO: sample from prior

    # initial weight
    weight = 
    current_pos = 
    for current_target in 1:num_steps
        # TODO: define new target using schedule

        # TODO: perform MCMC transition
        
        # - generate proposal
        prop = current_pos + rand(proposal)
        
        # - compute acceptance ratio
        # query MCMC kernel density
        q_for = logpdf(proposal, prop - current_pos)
        q_back = logpdf(proposal, current_pos - prop)
        # query tempered distribution at prop and current_pos
        tempered_current_pos = tempered_log_prob(current_pos, beta, ???????)
        tempered_prop = tempered_log_prob(prop, beta, ???????)
        # deduce acceptance ratio

        # - accept or reject 
        

        # TODO: update weight
    end
    return Transition(spl)
end

# B.3. sample_end! combines the individual weights to obtain final_logevidence, as in vanilla IS 

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # Calculate evidence using weight attribute of new transition struct
    spl.state.final_logevidence = logsumexp(map(x->x.weight, ts)) - log(N)
end


## C. If necessary, overload assume and observe

# don't know exactly how i'll proceed here...

## D. helper functions 

# generic problem: logjoint, loglikelihood, etc. modify a vi
# but sometimes I just want the output value so that i can operate over it, and then modify a vi
function tempered_log_prob(x, beta, ???????) # spl? model?
    logprior = 
    logjoint = 
end