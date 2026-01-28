###
### Particle Filtering and Particle MCMC Samplers.
###

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

struct TracedModel{V<:AbstractVarInfo,M<:Model,T<:Tuple,NT<:NamedTuple} <:
       AdvancedPS.AbstractTuringLibtaskModel
    model::M
    varinfo::V
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
    kwargs...,
)
    check_model && _check_model(model, sampler)
    error_if_threadsafe_eval(model)
    check_model_kwargs(model)
    # need to add on the `nparticles` keyword argument for `initialstep` to make use of
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

function check_model_kwargs(model::DynamicPPL.Model)
    if !isempty(model.defaults)
        # If there are keyword arguments, we need to check that the user has
        # accounted for this by overloading `might_produce`.
        might_produce = Libtask.might_produce(typeof((Core.kwcall, NamedTuple(), model.f)))
        if !might_produce
            io = IOBuffer()
            ctx = IOContext(io, :color => true)
            print(
                ctx,
                "Models with keyword arguments need special treatment to be used" *
                " with particle methods. Please run:\n\n",
            )
            printstyled(
                ctx, "    Turing.@might_produce($(model.f))"; bold=true, color=:blue
            )
            print(ctx, "\n\nbefore sampling from this model with particle methods.\n")
            error(String(take!(io)))
        end
    end
end

function Turing.Inference.initialstep(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::SMC,
    vi::AbstractVarInfo;
    nparticles::Int,
    discard_sample=false,
    kwargs...,
)
    check_model_kwargs(model)
    # Reset the VarInfo.
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())
    vi = DynamicPPL.empty!!(vi)

    # Create a new set of particles.
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, vi, AdvancedPS.TracedRNG(), true) for _ in 1:nparticles],
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
        DynamicPPL.ParamsWithStats(deepcopy(particle.model.f.varinfo), model, stats)
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
        DynamicPPL.ParamsWithStats(deepcopy(particle.model.f.varinfo), model, stats)
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

struct PGState
    vi::AbstractVarInfo
    rng::Random.AbstractRNG
end

get_varinfo(state::PGState) = state.vi

function Turing.Inference.initialstep(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::PG,
    vi::AbstractVarInfo;
    discard_sample=false,
    kwargs...,
)
    error_if_threadsafe_eval(model)
    check_model_kwargs(model)
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())

    # Create a new set of particles
    num_particles = spl.nparticles
    particles = AdvancedPS.ParticleContainer(
        [
            AdvancedPS.Trace(model, vi, AdvancedPS.TracedRNG(), true) for
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
        DynamicPPL.ParamsWithStats(deepcopy(_vi), model, (; logevidence=logevidence))
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
    # Reset the VarInfo before new sweep.
    vi = state.vi
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())

    # Create reference particle for which the samples will be retained.
    reference = AdvancedPS.forkr(AdvancedPS.Trace(model, vi, state.rng, false))

    # Create a new set of particles.
    num_particles = spl.nparticles
    x = map(1:num_particles) do i
        if i != num_particles
            return AdvancedPS.Trace(model, vi, AdvancedPS.TracedRNG(), true)
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
        DynamicPPL.ParamsWithStats(deepcopy(_vi), model, (; logevidence=logevidence))
    end

    return transition, PGState(_vi, newreference.rng)
end

"""
    get_trace_local_varinfo()

Get the varinfo stored in the 'taped globals' of a `Libtask.TapedTask`.
"""
function get_trace_local_varinfo()
    trace = try
        Libtask.get_taped_globals(Any).other
    catch e
        e == KeyError(:task_variable) ? nothing : rethrow(e)
    end
    trace === nothing && error("No trace found in Libtask globals; this should not happen.")
    return trace.model.f.varinfo::AbstractVarInfo
end

"""
    get_trace_local_resampled()

Get the `resample` flag stored in the 'taped globals' of a `Libtask.TapedTask`.

This indicates whether new variable values should be sampled from the prior or not. For
example, in SMC, this is true for all particles; in PG, this is true for all particles
except the reference particle, whose trajectory must be reproduced exactly.
"""
function get_trace_local_resampled()
    trace = try
        Libtask.get_taped_globals(Any).other
    catch e
        e == KeyError(:task_variable) ? nothing : rethrow(e)
    end
    trace === nothing && error("No trace found in Libtask globals; this should not happen.")
    return trace.model.f.resample::Bool
end

"""
    get_trace_local_rng()

Get the `Trace` local rng if one exists.

If executed within a `TapedTask`, return the `rng` stored in the "taped globals" of the
task, otherwise return `vi`.
"""
function get_trace_local_rng()
    return Libtask.get_taped_globals(Any).rng
end

"""
    set_trace_local_varinfo(vi::AbstractVarInfo)

Set the `varinfo` stored in Libtask's taped globals. The 'other' taped global in Libtask
is expected to be an `AdvancedPS.Trace`.

Returns `nothing`.
"""
function set_trace_local_varinfo(vi::AbstractVarInfo)
    trace = Libtask.get_taped_globals(Any).other
    trace === nothing && error("No trace found in Libtask globals; this should not happen.")
    model = trace.model
    model = Accessors.@set model.f.varinfo = vi
    trace.model = model
    return nothing
end

function DynamicPPL.tilde_assume!!(
    ctx::ParticleMCMCContext, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    # Get all the info we need from the trace, namely, the stored VarInfo, and whether
    # we need to sample a new value or use the existing one.
    vi = get_trace_local_varinfo()
    trng = get_trace_local_rng()
    resample = get_trace_local_resampled()
    # Modify the varinfo as appropriate.
    dispatch_ctx = if ~haskey(vi, vn) || resample
        DynamicPPL.InitContext(trng, DynamicPPL.InitFromPrior())
    else
        DynamicPPL.DefaultContext()
    end
    x, vi = DynamicPPL.tilde_assume!!(dispatch_ctx, dist, vn, vi)
    # Set the varinfo back in the trace.
    set_trace_local_varinfo(vi)
    return x, vi
end

function DynamicPPL.tilde_observe!!(
    ::ParticleMCMCContext,
    right::Distribution,
    left,
    vn::Union{VarName,Nothing},
    vi::AbstractVarInfo,
)
    vi = get_trace_local_varinfo()
    left, vi = DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, vi)
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
    acc::ProduceLogLikelihoodAccumulator, val, logjac, vn, right
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ProduceLogLikelihoodAccumulator, right, left, vn
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
Libtask.@might_produce(DynamicPPL.evaluate!!)
Libtask.@might_produce(DynamicPPL.init!!)
Libtask.might_produce(::Type{<:Tuple{<:DynamicPPL.Model,Vararg}}) = true
