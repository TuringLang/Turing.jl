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
    proposal_kernels # :: Array{AdvancedMH.MetropolisHastings}
    "array of inverse temperatures"
    schedule #  :: Array{<:AbstractFloat}
end

# TODO: add default constructor both for schedule (maybe 0.1:0.1:0.9?) and proposals (maybe Normals all the way, problem = dimension)

# A.2. state: similar to vanilla IS, with densitymodels for intermediate distributions added

"""
    AISState{V<:VarInfo, F<:AbstractFloat}

State struct for AIS: contains information about intermediate distributions and proposal kernels that are needed for all particles.
"""
mutable struct AISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    "varinfo - reset and computed in step!"
    vi                 ::  V # reset for every step ie particle
    "list of density models corresponding to intermediate target distributions, ending with the logjoint density model - computed in sample_init!"
    densitymodels      #:: Array{AdvancedMH.DensityModel}
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

# TODO: modify to memorize full path

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

    # evaluate logjoint at current_state
    lp = logdensity(last(spl.alg.densitymodels), current_state)
    
    # add lp as final term to accum_logweight
    accum_logweight += lp

    return AISTransition(current_state, lp, accum_logweight)
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

# TODO: make current_state and proposed_state NamedTuples
# propose() returns an AdvancedMH.Transition, which has a params field of type T<:Union{Vector, Real, NamedTuple} (in this case, NamedTuple)
# to change: 
# - make prior_step generate a NamedTuple
# - make proposed_state = propose(...).params a NamedTuple
# - make logdensities apply to NamedTuples rather than just arrays

# TypedVarInfo -> NamedTuple: via tonamedtuple
# output namedtuple has keys which are latent variable names 
# for each variable name, the corresponding value is a tuple containing a) the array of associated values b) the strings for the names of the individual components inside the variable

# question: am i dealing with Typed or Untyped varinfos here?
# for sampling from prior, untyped. in which case I have to build the 

# question: what do I keep, what do I change?


"""
    gen_logjoint(v, model, spl)

Return the log joint density function corresponding to model.
"""
function gen_logjoint(v, model, spl)
    function logjoint(z)::Float64
        z_old, lj_old = v[spl], getlogp(v) # TODO: figure out what spl is here
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
        z_old, lj_old = v[spl], getlogp(v) # TODO: figure out what spl is here
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
    prior_step(model)

Sample from prior to return inital values of current_state and accum_logweight.
"""
# TODO: make current_state a NamedTuple
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

"""
    intermediate_step(j, spl, current_state, accum_logweight)

Perform the MCMC step corresponding to the j-th intermediate distribution, with the j-th MH proposal. Return updated current_state and accum_logweight.
"""
function intermediate_step(j, spl, current_state, accum_logweight)
    # fetch proposal_kernel and densitymodel for this intermediate step
    densitymodel = spl.state.densitymodels[j]
    proposal_kernel = spl.alg.proposal_kernels[j]
    
    # generate new proposal: this is a Transition...
    proposed_state = AdvancedMH.propose(rng, proposal_kernel, densitymodel, current_state)

    # compute difference between intermediate logdensity at proposed and current positions
    diff_logdensity = AdvancedMH.logdensity(densitymodel, proposed_state) - AdvancedMH.logdensity(densitymodel, current_state)

    # calculate log acceptance probability.
    logα =  diff_logdensity + AdvancedMH.q(proposal_kernel, current_state, proposed_state) - AdvancedMH.q(proposal_kernel, proposed_state, current_state)

    # decide whether to accept or reject proposal
    if -Random.randexp(rng) < logα
        # accept: update current_state and accum_logweight
        accum_logweight -= diff_logdensity
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