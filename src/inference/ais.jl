## A. Sampler

# A.1. algorithm

""" 
    AIS

Simple version of AIS (not fully general).

Contains:
- intermediate distributions that come from tempering according to a schedule
- user-specified list of proposal Markov kernels
- MCMC acceptance ratios enforce invariance of kernels wrt intermediate distributions
"""
struct AIS <: InferenceAlgorithm 
    "array of intermediate MH kernels"
    proposal_kernels # :: Array{<:AdvancedMH.RandomWalkProposal} TODO: fix
    "array of inverse temperatures"
    schedule # :: Array{<:AbstractFloat} TODO: fix
end

DynamicPPL.getspace(::AIS) = ()

# TODO: add default constructor both for schedule (maybe 0.1:0.1:0.9?) and proposals (eg given a centered Gaussian vector's covariance matrix - this probably exists in AdvancedMh)

# A.2. state: similar to vanilla IS, with densitymodels for intermediate distributions added

"""
    AISState{V<:VarInfo, F<:AbstractFloat}

State struct for AIS: contains information about intermediate distributions and proposal kernels that are needed for all particles.
"""
mutable struct AISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    "varinfo - reset and computed in step!"
    vi                 ::  V # reset for every step ie particle
    "list of density models corresponding to intermediate target distributions, ending with the logjoint density model - computed in sample_init!"
    densitymodels      # :: Array{<:AdvancedMH.DensityModel} TODO: fix
    "log of the average of the particle weights: estimator of the log evidence - computed in sample_end!"
    final_logevidence  ::  F
end

AISState(model::Model) = AISState(VarInfo(model), [], 0.0)

# A.3. Sampler constructor: same as for vanilla IS

"""
    Sampler(alg::AIS, model::Model, s::Selector)

Sampler constructor: similar to vanilla IS.
"""
function Sampler(alg::AIS, model::Model, s::Selector)
    @assert length(alg.schedule) == length(alg.proposal_kernels)
    info = Dict{Symbol, Any}()
    state = AISState(model)
    return Sampler(alg, info, s, state)
end

## B. Implement AbstractMCMC

# B.1. new transition type AISTransition, with an additional attribute accum_logweight

"""
    AISTransition{T, F<:AbstractFloat}

AIS-specific Transition struct. 

Necessary because we care both about a particle's weight (accum_logweight) and the logjoint density evaluated at its final position (lp).
"""
struct AISTransition{T, F<:AbstractFloat}
    "parameter"
    θ  :: T
    "logjoint evaluated at θ"
    lp :: F
    "logarithm of the particle's AIS weight - accumulated during annealing run"
    accum_logweight :: F
end

# idk what this function is for
function additional_parameters(::Type{<:AISTransition})
    return [:lp, :accum_logweight]
end


# B.2. sample_init! function 

"""
    AbstractMCMC.sample_init!

Initialize AISState attributes densitymodels and logjoint in spl.state.
"""
function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    logjoint = gen_logjoint(spl.state.vi, model, spl)
    logprior = gen_logprior(spl.state.vi, model, spl)

    spl.state.densitymodels = [] 

    # densitymodels for intermediate distributions
    for beta in spl.alg.schedule
        log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
        densitymodel = AdvancedMH.DensityModel(log_unnorm_tempered)
        push!(spl.state.densitymodels, densitymodel)
    end

    # densitymodel for the logjoint, ie the final target
    final_densitymodel = AdvancedMH.DensityModel(logjoint)
    push!(spl.state.densitymodels, final_densitymodel)
end

# B.3. step function 

# TODO: memorize full path: not much to change, but check what we care about exactly
# - all proposals?
# - all values of accum_logweight?
# - all values of diff_logdensity?
# - all values of f_j(x_j)?

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    ::Integer,
    transition;
    kwargs...
)
    # sample from prior and compute first term in accum_logweight
    current_state, accum_logweight = prior_step(spl, model)

    # for every intermediate distribution
    for j in 1:length(spl.alg.schedule)
        current_state, accum_logweight = intermediate_step(j, spl, current_state, accum_logweight)
    end

    # evaluate logjoint at current_state
    lp = AdvancedMH.logdensity(last(spl.state.densitymodels), current_state)
    
    # add lp as final term to accum_logweight
    accum_logweight += lp

    # update spl to set the path VarInfo
    spl.state.vi[spl] = current_state

    # use path VarInfo to build instance of AISTransition
    nt = NamedTuple()
    theta = merge(DynamicPPL.tonamedtuple(spl.state.vi), NamedTuple())
    return AISTransition(theta, lp, accum_logweight)
end

# B.4. sample_end! combines the individual accum_logweights to obtain final_logevidence, as in vanilla IS 

"""
    AbstractMCMC.sample_end!( ::AbstractRNG, ::Model, spl::Sampler{<:AIS}, N::Integer, ts::Vector; kwargs...)

Store estimate of the log evidence in the AISState attribute final_logevidence of spl.state.
"""
function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:AIS},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # use AISTransition accum_logweight attribute to store estimate of log evidence
    spl.state.final_logevidence = logsumexp(map(x->x.accum_logweight, ts)) - log(N)
end

# C. helper functions


"""
    gen_logjoint(v, model, spl)

Return the log joint density function corresponding to model.
"""
function gen_logjoint(v, model, spl)
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

"""
    gen_logprior(v, model, spl)

Return the log prior density function corresponding to model.
"""
function gen_logprior(v, model, spl)
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

"""
    gen_log_unnorm_tempered(logprior, logjoint, beta)

Return the log unnormalized tempered density function corresponding to model, ie a convex combination of the logprior and logjoint densities with parameter beta.
"""
function gen_log_unnorm_tempered(logprior, logjoint, beta)
    function log_unnorm_tempered(z)
        return (1 - beta) * logprior(z) + beta * logjoint(z)
    end
    return log_unnorm_tempered
end

"""
    prior_step(spl, model)

Sample from prior to return inital values of current_state and accum_logweight.
"""
function prior_step(spl, model)
    # sample from prior
    prior_vi = VarInfo()
    prior_spl = SampleFromPrior()
    model(prior_vi, prior_spl)

    # initialize current_state
    current_state = prior_vi[prior_spl]

    # initialize accum_logweight
    logprior = gen_logprior(spl.state.vi, model, spl)
    accum_logweight = - logprior(current_state)

    return current_state, accum_logweight
end

"""
    intermediate_step(j, spl, current_state, accum_logweight)

Perform the MCMC step corresponding to the j-th intermediate distribution, with the j-th MH proposal. Return updated current_state and accum_logweight.
"""
function intermediate_step(j, spl, current_state, accum_logweight)
    # fetch proposal_kernel and densitymodel for this intermediate step
    densitymodel = spl.state.densitymodels[j]
    proposal_kernel = spl.alg.proposal_kernels[j]
    
    # TODO: generalize - for now, proposal_kernel can only be a RandomWalkProposal
    proposed_state = current_state + rand(proposal_kernel)

    # compute difference between intermediate logdensity at proposed and current positions
    diff_logdensity = AdvancedMH.logdensity(densitymodel, proposed_state) - AdvancedMH.logdensity(densitymodel, current_state)

    # calculate log acceptance probability.
    logα =  diff_logdensity + AdvancedMH.q(proposal_kernel, current_state, proposed_state) - AdvancedMH.q(proposal_kernel, proposed_state, current_state)

    # decide whether to accept or reject proposal
    if -Random.randexp() < logα
        # accept: update current_state and accum_logweight
        accum_logweight -= diff_logdensity
        current_state = proposed_state
    # reject: no updates necessary
    end

    return current_state, accum_logweight
end

## D. overload assume and observe: similar to hmc.jl, so that gen_logjoint and gen_logprior work

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