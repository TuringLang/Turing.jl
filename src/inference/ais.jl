# TODO: fix convention to refer to MCMC steps within a transition, and independent AISTransition transitions ie particles...
# TODO: ensure correct typing in samplers, states, etc.
## A. Sampler

# A.1. algorithm

# simple version of AIS (not fully general): 
# - sequence of tempered distributions
# - list of proposal Markov kernels
# - MCMC acceptance ratios enforce invariance of kernels wrt intermediate distributions

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
    vi                 ::  V # reset for every step ie particle
    "log of the average of the particle weights: estimator of the log evidence"
    final_logevidence  ::  F
    "list of density models corresponding to intermediate target distributions - computed in sample_init!"
    list_densitymodels :: Array{DensityModel} # TODO: check type here
    "list of intermediate MH kernels"
    list_mh_samplers :: Array{MetropolisHastings} # TODO: check type here
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


# B.2. sample_init! function: initializes the AISState attributes list_densitymodels and list_mh_samplers # TODO: perhaps this could be done even earlier, in the AISState constructor?

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:AIS},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    spl.state.list_densitymodels = []
    spl.state.list_mh_samplers = []
    log_prior = gen_log_prior(spl.state.vi, model)
    log_joint = gen_log_joint(spl.state.vi, model)
    for i in 1:length(spl.alg.proposals)
        beta = spl.alg.schedule[i]
        log_unnorm_tempered = gen_log_unnorm_tempered(log_prior, log_joint, beta)
        densitymodel = AdvancedMH.DensityModel(log_unnorm_tempered)
        append!(spl.state.list_densitymodels, densitymodel)
        
        proposal = spl.alg.proposals[i]
        mh_sampler = AMH.MetropolisHastings(proposal) # maybe use RWMH(d) with d the associated distribution
        append!(spl.state.list_mh_samplers, mh_sampler)
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
    empty!(spl.state.vi) # particles are independent: previous step doesn't matter
    
    # sample from prior to initialize current_state
    prior_vi = VarInfo()
    prior_spl = SampleFromPrior()
    model(prior_vi, prior_spl)
    current_state = prior_vi[prior_spl]

    # initialize accum_logweight as minus log the prior evaluated at the sample
    accum_logweight = - log_prior(current_state)

    # for every intermediate distribution:
    for j in 1:length(spl.alg.schedule)
        # mh_sampler and densitymodel for this intermediate step
        densitymodel = spl.state.list_densitymodels[j]
        mh_sampler = spl.state.list_mh_samplers[j]
        
        # Generate a new proposal.
        proposed_state = propose(rng, mh_sampler, densitymodel, current_state)

        # Calculate the log acceptance probability.
        logα = densitymodel.logdensity(model, proposed_state) - densitymodel.logdensity(model, current_state) +
            q(mh_sampler, current_state, proposed_state) - q(mh_sampler, proposed_state, current_state)

        # Decide whether to accept or reject proposal
        if -Random.randexp(rng) < logα
            # TODO: accept: update current_state and accum_logweight
        else
            # TODO: reject: update current_state and accum_logweight
        end
    end
    # do a last accum_logweight update
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


## C. overload assume and observe: similar to MH, so that gen_log_joint and gen_log_prior work

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

# D. helper functions


function gen_log_joint(v, model)
    function log_joint(z)::Float64
        z_old, lj_old = v[spl], getlogp(v)
        v[spl] = z
        model(v, spl)
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return log_joint
end

function gen_log_prior(v, model)
    function log_prior(z)::Float64
        z_old, lj_old = v[spl], getlogp(v)
        v[spl] = z
        model(v, SampleFromPrior(), PriorContext())
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return log_prior
end

function gen_log_unnorm_tempered(log_prior, log_joint, beta)
    function log_unnorm_tempered(z)
        return (1 - beta) * log_prior(z) + beta * log_joint(z)
    end
    return log_unnorm_tempered
end

