# TODO: ensure correct typing in samplers, states, etc.
## A. Sampler

# A.1. algorithm

""" simple version of AIS (not fully general): 
- sequence of tempered distributions
- list of proposal Markov kernels
- MCMC acceptance ratios enforce invariance of kernels wrt intermediate distributions
"""
struct AIS <: InferenceAlgorithm 
    "array of `num_steps` AdvancedMH proposals"
    proposals :: Array{<:Proposal{P}}
    "array of `num_steps` inverse temperatures"
    schedule :: Array{<:Integer}
end

# TODO: add default constructor both for schedule and proposals
# TODO: distinguish static and dynamic proposals

# A.2. state: similar to vanilla IS, with intermediate density models and MC proposals added

mutable struct AISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    "varinfo - reset and computed in step!"
    vi                 ::  V # reset for every step ie particle
    "log of the average of the particle weights: estimator of the log evidence - computed in sample_end!"
    final_logevidence  ::  F
    "list of density models corresponding to intermediate target distributions - computed in sample_init!"
    densitymodels :: Array{DensityModel} # TODO: check type here
    "list of intermediate MH kernels - computed in sample_init!"
    proposal_kernels :: Array{MetropolisHastings} # TODO: check type here
    "log joint density - computed in sample_init!"
    logjoint :: G # TODO: check type here, this is the function rather than the density model
end

AISState(model::Model) = AISState(VarInfo(model), 0.0)

# A.3. Sampler constructor: same as for vanilla IS

function Sampler(alg::AIS, model::Model, s::Selector)
    @assert length(alg.schedule) == length(alg.proposals)
    info = Dict{Symbol, Any}()
    state = AISState(model)
    return Sampler(alg, info, s, state)
end


## B. Implement AbstractMCMC

# each time we call step!, we create a new particle as a transition like in is.jl

# B.1. new transition type AISTransition, with an additional attribute accum_logweight

struct AISTransition{T, F<:AbstractFloat}
    "parameter"
    θ  :: T
    "logjoint evaluated at θ"
    lp :: F
    "logarithm of the particle's AIS weight - accumulated during annealing run"
    accum_logweight :: F
end

function AISTransition(spl::Sampler, accum_logweight::F<:AbstractFloat, nt::NamedTuple=NamedTuple())
    theta = merge(tonamedtuple(spl.state.vi), nt)
    lp = getlogp(spl.state.vi)
    return AISTransition{typeof(theta), typeof(lp)}(theta, lp, accum_logweight)
end

# idk what this function is for
function additional_parameters(::Type{<:AISTransition})
    return [:lp, :accum_logweight]
end


# B.2. sample_init! function: initializes the AISState attributes densitymodels and proposal_kernels 

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    spl.state.densitymodels = []
    spl.state.proposal_kernels = []
    logjoint = gen_logjoint(spl.state.vi, model)
    spl.state.logjoint = logjoint
    
    logprior = gen_logprior(spl.state.vi, model)
    for i in 1:length(spl.alg.proposals)
        beta = spl.alg.schedule[i]
        log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
        densitymodel = AdvancedMH.DensityModel(log_unnorm_tempered)
        append!(spl.state.densitymodels, densitymodel)
        
        proposal = spl.alg.proposals[i]
        proposal_kernel = AMH.MetropolisHastings(proposal) # maybe use RWMH(d) with d the associated distribution
        append!(spl.state.proposal_kernels, proposal_kernel)
    end
end

# B.3. step function 
# TODO: should we memorize full path?
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    ::Integer,
    transition;
    kwargs...
)
    # particles are independent: forget previous particle's VarInfo
    empty!(spl.state.vi)

    # sample from prior and compute first term in accum_logweight
    current_state, accum_logweight = prior_step(model)

    # for every intermediate distribution
    for j in 1:length(spl.alg.schedule)
        current_state, accum_logweight = intermediate_step(j, spl, current_state, accum_logweight)
    end

    # add final term to accum_logweight
    accum_logweight += spl.state.logjoint(current_state)
end

# B.4. sample_end! combines the individual accum_logweights to obtain final_logevidence, as in vanilla IS 

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:IS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # update AISTransition accum_logweight attribute to store estimate of log evidence
    spl.state.final_logevidence = logsumexp(map(x->x.accum_logweight, ts)) - log(N)
end

# C. helper functions

function gen_logjoint(v, model)
    function logjoint(z)::Float64
        z_old, lj_old = v[spl], getlogp(v)
        v[spl] = z
        model(v, spl)
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return logjoint
end

function gen_logprior(v, model)
    function logprior(z)::Float64
        z_old, lj_old = v[spl], getlogp(v)
        v[spl] = z
        model(v, SampleFromPrior(), PriorContext())
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return logprior
end

function gen_log_unnorm_tempered(logprior, logjoint, beta)
    function log_unnorm_tempered(z)
        return (1 - beta) * logprior(z) + beta * logjoint(z)
    end
    return log_unnorm_tempered
end

function prior_step(model)
    # sample from prior
    prior_vi = VarInfo()
    prior_spl = SampleFromPrior()
    model(prior_vi, prior_spl)

    # initialize current_state
    current_state = prior_vi[prior_spl]

    # initialize accum_logweight
    accum_logweight = - logprior(current_state)

    return current_state, accum_logweight
end

function intermediate_step(j, spl, current_state, accum_logweight)
    # fetch proposal_kernel and densitymodel for this intermediate step
    densitymodel = spl.state.densitymodels[j]
    proposal_kernel = spl.state.proposal_kernels[j]
    
    # generate new proposal
    proposed_state = propose(rng, proposal_kernel, densitymodel, current_state)

    # compute difference between intermediate logdensity at proposed and current positions
    diff_logdensity = logdensity(densitymodel, proposed_state) - logdensity(densitymodel, current_state)

    # calculate log acceptance probability.
    logα =  diff_logdensity + q(proposal_kernel, current_state, proposed_state) - q(proposal_kernel, proposed_state, current_state)

    # decide whether to accept or reject proposal
    if -Random.randexp(rng) < logα
        # accept: update current_state and accum_logweight
        accum_logweight -= diff_logdensity(densitymodel, current_state)
        current_state = proposed_state
    # reject: no updates necessary
    end

    return current_state, accum_logweight
end

## D. overload assume and observe: similar to MH, so that gen_logjoint and gen_logprior work

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:AIS},
    dist::Distribution,
    vn::VarName,
    vi,
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:AIS},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi,
)
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:AIS},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function DynamicPPL.observe(
    spl::Sampler{<:AIS},
    d::Distribution,
    value,
    vi,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:AIS},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end