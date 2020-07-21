# TODO: fix convention to refer to MCMC steps within a transition, and independent AISTransition transitions ie particles...

## A. Sampler

# A.1. algorithm

# simple version of AIS (not fully general): 
# - sequence of distributions defined by tempering
# - transition kernels are MCMC kernels with the same proposal
# - acceptance ratios vary to ensure invariance

# TODO: maybe do something for schedule here, maybe not

struct AIS <: InferenceAlgorithm 
    num_steps :: Integer # number of MCMC steps = number of intermediate distributions + 1
    proposal :: Distributions.Distribution # must be able to sample AND to compute densities (technically, only the former if symmetric)
    schedule :: Array{<:Integer,1} # probably a list - that must contain num_steps elements
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

# TODO: decide how to handle multiple particles. a few possibilities:
# 1. preferred: each time we call step!, we create a new particle as a transition like in is.jl (how does is.jl handle parallelization?)
# 2. maybe: this whole file focuses on a single particle, parallelization handled at a higher level (problem: combining particles to estimate log-evidence?)
# 3. maybe: something more like in advancedPS (more complex, doesn't seem necessary?)

# B.1. new transition type AISTransition, with an additional attribute logweight

struct AISTransition{T, F<:AbstractFloat}
    θ  :: T
    lp :: F
    logweight :: F
end

function AISTransition(spl::Sampler, logweight::F<:AbstractFloat, nt::NamedTuple=NamedTuple())
    theta = merge(tonamedtuple(spl.state.vi), nt)
    lp = getlogp(spl.state.vi)
    return AISTransition{typeof(theta), typeof(lp)}(theta, lp, logweight)
end

# idk what this function is for
function additional_parameters(::Type{<:AISTransition})
    return [:lp, :logweight]
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
    
    # TODO: sample from prior - but I don't want to add it to vi? should I create a new artificial vi?

    # initial logweight
    logweight = 
    current_pos = 
    for current_target in 1:num_steps
        # TODO: define new target using schedule

        # TODO: perform MCMC transition
        
        # generate proposal
        prop = current_pos + rand(proposal)
        
        # compute acceptance ratio
        # query MCMC kernel density
        T_forward = logpdf(proposal, prop - current_pos)
        T_backward = logpdf(proposal, current_pos - prop) # this is NOT \tilde{T} from the AIS paper
        # query tempered distribution at prop and current_pos
        tempered_current_pos = tempered_log_prob(current_pos, beta, ???????)
        tempered_prop = tempered_log_prob(prop, beta, ???????)
        # deduce acceptance ratio
        ratio = min(1, exp(tempered_prop + T_backward - tempered_current_pos - T_forward))
        
        # accept or reject: if accept update current_pos and logweight, if reject both stay the same
        if rand() < ratio
            logweight += tempered_current_pos - tempered_prop
            current_pos = prop
        end
    end

    # final logweight update
    logweight += # evaluate log joint density at current_pos
    return AISTransition(spl)
end

# B.3. sample_end! combines the individual logweights to obtain final_logevidence, as in vanilla IS 

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # use AISTransition logweight attribute
    spl.state.final_logevidence = logsumexp(map(x->x.logweight, ts)) - log(N)
end


## C. If necessary, overload assume and observe

# don't know exactly how i'll proceed here...

## D. helper functions 

# generic problem: logjoint, loglikelihood, etc. modify a vi
# but sometimes I just want the output value so that i can operate over it, and then modify a vi
function tempered_log_prob(x, beta, spl)
    logprior = 
    logjoint = 
end