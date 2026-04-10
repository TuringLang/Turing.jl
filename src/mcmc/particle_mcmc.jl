###
### Particle Filtering and Particle MCMC Samplers.
###

using Accessors: Accessors

function error_if_threadsafe_eval(model::DynamicPPL.Model)
    if DynamicPPL.requires_threadsafe(model)
        throw(
            ArgumentError(
                "Particle sampling methods do not currently support models that need threadsafe evaluation.",
            ),
        )
    end
    return nothing
end

### AdvancedPS models and interface

struct ParticleMCMCContext{R<:AbstractRNG} <: DynamicPPL.AbstractContext
    rng::R
end
# Because pMCMC uses OnlyAccsVarInfo, we need to overload this. It's fine to use Any (see
# the docstring of get_param_eltype in DynamicPPL) because pMCMC doesn't involve AD or any
# other tracer types.
DynamicPPL.get_param_eltype(::DynamicPPL.AbstractVarInfo, ::ParticleMCMCContext) = Any

mutable struct TracedModel{M<:Model,T<:Tuple,NT<:NamedTuple} <:
               AdvancedPS.AbstractGenericModel
    model::M
    # TODO(penelopeysm): I don't like that this is an abstract type. However, the problem is
    # that the type of VarInfo can change during execution, especially with PG-inside-Gibbs
    # when you have to muck with merging VarInfos from different sub-conditioned models.
    #
    # However, I don't think that this is actually a problem in practice. Whenever we do
    # Libtask.get_taped_globals, that is already type unstable anyway, so accessing this
    # field here is not going to cause extra type instability. This change is associated
    # with Turing v0.43, and I benchmarked on v0.42 vs v0.43, and v0.43 is actually faster
    # (probably due to underlying changes in DynamicPPL), so I'm not really bothered by
    # this.
    varinfo::AbstractVarInfo
    resample::Bool
    fargs::T
    kwargs::NT
end

function TracedModel(
    model::Model, varinfo::AbstractVarInfo, rng::Random.AbstractRNG, resample::Bool
)
    model = DynamicPPL.setleafcontext(model, ParticleMCMCContext(rng))
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    fargs = (model.f, args...)
    return TracedModel(model, varinfo, resample, fargs, kwargs)
end

function AdvancedPS.advance!(
    trace::AdvancedPS.Trace{<:AdvancedPS.LibtaskModel{<:TracedModel}}, isref::Bool=false
)
    # Make sure we load/reset the rng in the new replaying mechanism
    isref ? AdvancedPS.load_state!(trace.rng) : AdvancedPS.save_state!(trace.rng)
    score = consume(trace.model.ctask)
    return score
end

function AdvancedPS.delete_retained!(trace::TracedModel)
    # This method is called if, during a CSMC update, we perform a resampling
    # and choose the reference particle as the trajectory to carry on from.
    # In such a case, we need to ensure that when we continue sampling (i.e.
    # the next time we hit tilde_assume!!), we don't use the values in the 
    # reference particle but rather sample new values.
    return TracedModel(trace.model, trace.varinfo, true, trace.fargs, trace.kwargs)
end

function AdvancedPS.reset_model(trace::TracedModel)
    return trace
end

function Libtask.TapedTask(taped_globals, model::TracedModel)
    return Libtask.TapedTask(
        taped_globals, model.fargs[1], model.fargs[2:end]...; model.kwargs...
    )
end

abstract type ParticleInference <: AbstractSampler end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
$(TYPEDEF)

Sequential Monte Carlo sampler.

# Fields

$(TYPEDFIELDS)
"""
struct SMC{R} <: ParticleInference
    resampler::R
end

"""
    SMC([resampler = AdvancedPS.ResampleWithESSThreshold()])
    SMC([resampler = AdvancedPS.resample_systematic, ]threshold)

Create a sequential Monte Carlo sampler of type [`SMC`](@ref).

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
SMC() = SMC(AdvancedPS.ResampleWithESSThreshold())

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real)
    return SMC(AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function SMC(threshold::Real)
    return SMC(AdvancedPS.resample_systematic, threshold)
end

struct SMCState{P,F<:AbstractFloat}
    particles::P
    particleindex::Int
    # The logevidence after aggregating all samples together.
    average_logevidence::F
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    N::Integer;
    check_model=true,
    chain_type=DEFAULT_CHAIN_TYPE,
    initial_params=Turing.Inference.init_strategy(sampler),
    progress=PROGRESS[],
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    error_if_threadsafe_eval(model)
    # SMC does not produce a Markov chain, so discard_initial and thinning do not apply.
    # We consume these keyword arguments here to prevent them from being passed to
    # AbstractMCMC.mcmcsample, which would cause a BoundsError (#1811).
    if discard_initial > 0 || thinning > 1
        @warn "SMC samplers do not support `discard_initial` or `thinning`. These keyword arguments will be ignored."
    end
    return AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        chain_type=chain_type,
        initial_params=initial_params,
        progress=progress,
        nparticles=N,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::SMC;
    nparticles::Int,
    initial_params,
    discard_sample=false,
    kwargs...,
)
    # Create an empty VarInfo
    accs = DynamicPPL.OnlyAccsVarInfo()
    accs = DynamicPPL.setacc!!(accs, ProduceLogLikelihoodAccumulator())
    accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(true))

    # Create a new set of particles.
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, accs, AdvancedPS.TracedRNG(), true) for _ in 1:nparticles],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.resampler, spl)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = AdvancedPS.getweight(particles, 1)

    # Compute the first transition and the first state.
    stats = (; weight=weight, logevidence=logevidence)
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(particle.model.f.varinfo, stats)
    end
    state = SMCState(particles, 2, logevidence)

    return transition, state
end

function AbstractMCMC.step(
    ::AbstractRNG,
    model::DynamicPPL.Model,
    spl::SMC,
    state::SMCState;
    discard_sample=false,
    kwargs...,
)
    # Extract the index of the current particle.
    index = state.particleindex

    # Extract the current particle and its weight.
    particles = state.particles
    particle = particles.vals[index]
    weight = AdvancedPS.getweight(particles, index)

    # Compute the transition and the next state.
    stats = (; weight=weight, logevidence=state.average_logevidence)
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(deepcopy(particle.model.f.varinfo), stats)
    end
    nextstate = SMCState(state.particles, index + 1, state.average_logevidence)

    return transition, nextstate
end

####
#### Particle Gibbs sampler.
####

"""
$(TYPEDEF)

Particle Gibbs sampler.

# Fields

$(TYPEDFIELDS)
"""
struct PG{R} <: ParticleInference
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

"""
PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold()])
PG(n, [resampler = AdvancedPS.resample_systematic, ]threshold)

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(nparticles::Int)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold())
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function PG(nparticles::Int, threshold::Real)
    return PG(nparticles, AdvancedPS.resample_systematic, threshold)
end

"""
CSMC(...)

Equivalent to [`PG`](@ref).
"""
const CSMC = PG # type alias of PG as Conditional SMC

struct PGState{V<:DynamicPPL.AbstractVarInfo,R<:Random.AbstractRNG}
    vi::V
    rng::R
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::PG; discard_sample=false, kwargs...
)
    error_if_threadsafe_eval(model)
    oavi = DynamicPPL.OnlyAccsVarInfo()
    oavi = DynamicPPL.setacc!!(oavi, ProduceLogLikelihoodAccumulator())
    oavi = DynamicPPL.setacc!!(oavi, DynamicPPL.RawValueAccumulator(true))

    # Create a new set of particles
    num_particles = spl.nparticles
    particles = AdvancedPS.ParticleContainer(
        [
            AdvancedPS.Trace(model, oavi, AdvancedPS.TracedRNG(), true) for
            _ in 1:num_particles
        ],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.resampler, spl)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    index = AdvancedPS.randcat(rng, Ws)
    reference = particles.vals[index]

    # Compute the first transition.
    _vi = reference.model.f.varinfo
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(deepcopy(_vi), (; logevidence=logevidence))
    end

    return transition, PGState(_vi, reference.rng)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::PG,
    state::PGState;
    discard_sample=false,
    kwargs...,
)
    # Reset log-prob accs in reference particle, to avoid accumulating into the same accs
    # across iterations. If the chosen particle for this iteration is the reference
    # particle, this allows us to just read off the log-probs from the accumulators,
    # without having to re-evaluate the model.
    reference_vi = state.vi
    reference_vi = DynamicPPL.setacc!!(reference_vi, ProduceLogLikelihoodAccumulator())
    reference_vi = DynamicPPL.setacc!!(reference_vi, DynamicPPL.LogPriorAccumulator())
    reference_vi = DynamicPPL.setacc!!(reference_vi, DynamicPPL.LogJacobianAccumulator())

    # Create reference particle for which the samples will be retained.
    reference = AdvancedPS.forkr(AdvancedPS.Trace(model, reference_vi, state.rng, false))

    # Create a new set of particles with newly emptied accs
    empty_accs = DynamicPPL.OnlyAccsVarInfo()
    empty_accs = DynamicPPL.setacc!!(empty_accs, ProduceLogLikelihoodAccumulator())
    empty_accs = DynamicPPL.setacc!!(empty_accs, DynamicPPL.RawValueAccumulator(true))
    num_particles = spl.nparticles
    x = map(1:num_particles) do i
        if i != num_particles
            return AdvancedPS.Trace(model, empty_accs, AdvancedPS.TracedRNG(), true)
        else
            return reference
        end
    end
    particles = AdvancedPS.ParticleContainer(x, AdvancedPS.TracedRNG(), rng)

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.resampler, spl, reference)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    index = AdvancedPS.randcat(rng, Ws)
    newreference = particles.vals[index]

    # Compute the transition.
    _vi = newreference.model.f.varinfo
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(deepcopy(_vi), (; logevidence=logevidence))
    end

    return transition, PGState(_vi, newreference.rng)
end

"""
    get_trace_local_varinfo()

Get the varinfo stored in the 'taped globals' of a `Libtask.TapedTask`. This function
is meant to be called from *inside* the TapedTask itself.
"""
function get_trace_local_varinfo()
    trace = Libtask.get_taped_globals(Any).other
    return trace.model.f.varinfo::AbstractVarInfo
end

"""
    get_trace_local_resampled()

Get the `resample` flag stored in the 'taped globals' of a `Libtask.TapedTask`.

This indicates whether new variable values should be sampled from the prior or not. For
example, in SMC, this is true for all particles; in PG, this is true for all particles
except the reference particle, whose trajectory must be reproduced exactly.

This function is meant to be called from *inside* the TapedTask itself.
"""
function get_trace_local_resampled()
    trace = Libtask.get_taped_globals(Any).other
    return trace.model.f.resample::Bool
end

"""
    get_trace_local_rng()

Get the RNG stored in the 'taped globals' of a `Libtask.TapedTask`, if one exists.

This function is meant to be called from *inside* the TapedTask itself.
"""
function get_trace_local_rng()
    return Libtask.get_taped_globals(Any).rng
end

"""
    set_trace_local_varinfo(vi::AbstractVarInfo)

Set the `varinfo` stored in Libtask's taped globals. The 'other' taped global in Libtask
is expected to be an `AdvancedPS.Trace`.

Returns `nothing`.

This function is meant to be called from *inside* the TapedTask itself.
"""
function set_trace_local_varinfo(vi::AbstractVarInfo)
    trace = Libtask.get_taped_globals(Any).other
    trace.model.f.varinfo = vi
    return nothing
end

function DynamicPPL.tilde_assume!!(
    ::ParticleMCMCContext, dist::Distribution, vn::VarName, template::Any, ::AbstractVarInfo
)
    # Get all the info we need from the trace, namely, the stored VarInfo, and whether
    # we need to sample a new value or use the existing one.
    vi = get_trace_local_varinfo()
    trng = get_trace_local_rng()
    resample = get_trace_local_resampled()
    # Modify the varinfo as appropriate.
    values = DynamicPPL.get_raw_values(vi)
    init_strat = if ~haskey(values, vn) || resample
        DynamicPPL.InitFromPrior()
    else
        DynamicPPL.InitFromParams(values, nothing)
    end
    ctx = DynamicPPL.InitContext(trng, init_strat, DynamicPPL.UnlinkAll())
    x, vi = DynamicPPL.tilde_assume!!(ctx, dist, vn, template, vi)
    # Set the varinfo back in the trace.
    set_trace_local_varinfo(vi)
    return x, vi
end

function DynamicPPL.tilde_observe!!(
    ::ParticleMCMCContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template::Any,
    vi::AbstractVarInfo,
)
    vi = get_trace_local_varinfo()
    left, vi = DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, template, vi)
    set_trace_local_varinfo(vi)
    return left, vi
end

# Convenient constructor
function AdvancedPS.Trace(
    model::Model, varinfo::AbstractVarInfo, rng::AdvancedPS.TracedRNG, resample::Bool
)
    newvarinfo = deepcopy(varinfo)
    tmodel = TracedModel(model, newvarinfo, rng, resample)
    newtrace = AdvancedPS.Trace(tmodel, rng)
    return newtrace
end

"""
ProduceLogLikelihoodAccumulator{T<:Real} <: AbstractAccumulator

Exactly like `LogLikelihoodAccumulator`, but calls `Libtask.produce` on change of value.

# Fields
$(TYPEDFIELDS)
"""
struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.LogProbAccumulator{T}
    "the scalar log likelihood value"
    logp::T
end

# Note that this uses the same name as `LogLikelihoodAccumulator`. Thus only one of the two
# can be used in a given VarInfo.
DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood
DynamicPPL.logp(acc::ProduceLogLikelihoodAccumulator) = acc.logp

function DynamicPPL.acclogp(acc1::ProduceLogLikelihoodAccumulator, val)
    # The below line is the only difference from `LogLikelihoodAccumulator`.
    Libtask.produce(val)
    return ProduceLogLikelihoodAccumulator(acc1.logp + val)
end

function DynamicPPL.accumulate_assume!!(
    acc::ProduceLogLikelihoodAccumulator, val, tval, logjac, vn, right, template
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ProduceLogLikelihoodAccumulator, right, left, vn, template
)
    return DynamicPPL.acclogp(acc, Distributions.loglikelihood(right, left))
end

# We need to tell Libtask which calls may have `produce` calls within them. In practice most
# of these won't be needed, because of inlining and the fact that `might_produce` is only
# called on `:invoke` expressions rather than `:call`s, but since those are implementation
# details of the compiler, we set a bunch of methods as might_produce = true. We start with
# adding to ProduceLogLikelihoodAccumulator, which is what calls `produce`, and go up the
# call stack.
Libtask.@might_produce(DynamicPPL.accloglikelihood!!)
function Libtask.might_produce(
    ::Type{
        <:Tuple{
            typeof(Base.:+),
            ProduceLogLikelihoodAccumulator,
            DynamicPPL.LogLikelihoodAccumulator,
        },
    },
)
    return true
end
Libtask.@might_produce(DynamicPPL.accumulate_observe!!)
Libtask.@might_produce(DynamicPPL.tilde_observe!!)
# Could tilde_assume!! have tighter type bounds on the arguments, namely a GibbsContext?
# That's the only thing that makes tilde_assume calls result in tilde_observe calls.
Libtask.@might_produce(DynamicPPL.tilde_assume!!)

# This handles all models and submodel evaluator functions (including those with keyword
# arguments). The key to this is realising that all model evaluator functions take
# DynamicPPL.Model as an argument, so we can just check for that. See
# https://github.com/TuringLang/Libtask.jl/issues/217.
Libtask.might_produce_if_sig_contains(::Type{<:DynamicPPL.Model}) = true

####
#### Gibbs interface
####

function gibbs_get_raw_values(state::PGState)
    return DynamicPPL.get_raw_values(state.vi)
end

function gibbs_update_state!!(
    ::PG, state::PGState, model::DynamicPPL.Model, global_vals::DynamicPPL.VarNamedTuple
)
    init_strat = DynamicPPL.InitFromParams(global_vals, nothing)
    new_vi = last(DynamicPPL.init!!(model, state.vi, init_strat, DynamicPPL.UnlinkAll()))
    return PGState(new_vi, state.rng)
end
