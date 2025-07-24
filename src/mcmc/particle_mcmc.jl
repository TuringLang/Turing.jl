###
### Particle Filtering and Particle MCMC Samplers.
###

### AdvancedPS models and interface

struct TracedModel{S<:AbstractSampler,V<:AbstractVarInfo,M<:Model,E<:Tuple} <:
       AdvancedPS.AbstractGenericModel
    model::M
    sampler::S
    varinfo::V
    evaluator::E
end

function TracedModel(
    model::Model,
    sampler::AbstractSampler,
    varinfo::AbstractVarInfo,
    rng::Random.AbstractRNG,
)
    spl_context = DynamicPPL.SamplingContext(rng, sampler, model.context)
    spl_model = DynamicPPL.contextualize(model, spl_context)
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(spl_model, varinfo)
    if kwargs !== nothing && !isempty(kwargs)
        error(
            "Sampling with `$(sampler.alg)` does not support models with keyword arguments. See issue #2007 for more details.",
        )
    end
    evaluator = (spl_model.f, args...)
    return TracedModel(spl_model, sampler, varinfo, evaluator)
end

function AdvancedPS.advance!(
    trace::AdvancedPS.Trace{<:AdvancedPS.LibtaskModel{<:TracedModel}}, isref::Bool=false
)
    # We want to increment num produce for the VarInfo stored in the trace. The trace is
    # mutable, so we create a new model with the incremented VarInfo and set it in the trace
    model = trace.model
    model = Accessors.@set model.f.varinfo = DynamicPPL.increment_num_produce!!(
        model.f.varinfo
    )
    trace.model = model
    # Make sure we load/reset the rng in the new replaying mechanism
    isref ? AdvancedPS.load_state!(trace.rng) : AdvancedPS.save_state!(trace.rng)
    score = consume(trace.model.ctask)
    return score
end

function AdvancedPS.delete_retained!(trace::TracedModel)
    DynamicPPL.set_retained_vns_del!(trace.varinfo)
    return trace
end

function AdvancedPS.reset_model(trace::TracedModel)
    return Accessors.@set trace.varinfo = DynamicPPL.reset_num_produce!!(trace.varinfo)
end

function Libtask.TapedTask(taped_globals, model::TracedModel; kwargs...)
    return Libtask.TapedTask(
        taped_globals, model.evaluator[1], model.evaluator[2:end]...; kwargs...
    )
end

abstract type ParticleInference <: InferenceAlgorithm end

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

struct SMCTransition{T,F<:AbstractFloat} <: AbstractTransition
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample (NOTE: does not work, always set to zero)."
    lp::F
    "The weight of the particle the sample was retrieved from."
    weight::F
end

function SMCTransition(model::DynamicPPL.Model, vi::AbstractVarInfo, weight)
    theta = getparams(model, vi)
    lp = DynamicPPL.getlogjoint(vi)
    return SMCTransition(theta, lp, weight)
end

metadata(t::SMCTransition) = (lp=t.lp, weight=t.weight)

DynamicPPL.getlogp(t::SMCTransition) = t.lp

struct SMCState{P,F<:AbstractFloat}
    particles::P
    particleindex::Int
    # The logevidence after aggregating all samples together.
    average_logevidence::F
end

function getlogevidence(samples, sampler::Sampler{<:SMC}, state::SMCState)
    return state.average_logevidence
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Sampler{<:SMC},
    N::Integer;
    chain_type=DynamicPPL.default_chain_type(sampler),
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    progress=PROGRESS[],
    kwargs...,
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type=chain_type,
            progress=progress,
            nparticles=N,
            kwargs...,
        )
    else
        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type,
            initial_state,
            progress=progress,
            nparticles=N,
            kwargs...,
        )
    end
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    vi::AbstractVarInfo;
    nparticles::Int,
    kwargs...,
)
    # Reset the VarInfo.
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())
    vi = DynamicPPL.reset_num_produce!!(vi)
    DynamicPPL.set_retained_vns_del!(vi)
    vi = DynamicPPL.resetlogp!!(vi)
    vi = DynamicPPL.empty!!(vi)

    # Create a new set of particles.
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG()) for _ in 1:nparticles],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = AdvancedPS.getweight(particles, 1)

    # Compute the first transition and the first state.
    transition = SMCTransition(model, particle.model.f.varinfo, weight)
    state = SMCState(particles, 2, logevidence)

    return transition, state
end

function AbstractMCMC.step(
    ::AbstractRNG, model::AbstractModel, spl::Sampler{<:SMC}, state::SMCState; kwargs...
)
    # Extract the index of the current particle.
    index = state.particleindex

    # Extract the current particle and its weight.
    particles = state.particles
    particle = particles.vals[index]
    weight = AdvancedPS.getweight(particles, index)

    # Compute the transition and the next state.
    transition = SMCTransition(model, particle.model.f.varinfo, weight)
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

struct PGTransition{T,F<:AbstractFloat} <: AbstractTransition
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample (NOTE: does not work, always set to zero)."
    lp::F
    "The log evidence of the sample."
    logevidence::F
end

struct PGState
    vi::AbstractVarInfo
    rng::Random.AbstractRNG
end

varinfo(state::PGState) = state.vi

function PGTransition(model::DynamicPPL.Model, vi::AbstractVarInfo, logevidence)
    theta = getparams(model, vi)
    lp = DynamicPPL.getlogjoint(vi)
    return PGTransition(theta, lp, logevidence)
end

metadata(t::PGTransition) = (lp=t.lp, logevidence=t.logevidence)

DynamicPPL.getlogp(t::PGTransition) = t.lp

function getlogevidence(samples, sampler::Sampler{<:PG}, state::PGState)
    return mean(x.logevidence for x in samples)
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    vi::AbstractVarInfo;
    kwargs...,
)
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())
    # Reset the VarInfo before new sweep
    vi = DynamicPPL.reset_num_produce!!(vi)
    DynamicPPL.set_retained_vns_del!(vi)
    vi = DynamicPPL.resetlogp!!(vi)

    # Create a new set of particles
    num_particles = spl.alg.nparticles
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG()) for _ in 1:num_particles],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.model.f.varinfo
    transition = PGTransition(model, _vi, logevidence)

    return transition, PGState(_vi, reference.rng)
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::AbstractModel, spl::Sampler{<:PG}, state::PGState; kwargs...
)
    # Reset the VarInfo before new sweep.
    vi = state.vi
    vi = DynamicPPL.reset_num_produce!!(vi)
    vi = DynamicPPL.resetlogp!!(vi)

    # Create reference particle for which the samples will be retained.
    reference = AdvancedPS.forkr(AdvancedPS.Trace(model, spl, vi, state.rng))

    # For all other particles, do not retain the variables but resample them.
    for vn in keys(vi)
        DynamicPPL.set_flag!(vi, vn, "del")
    end

    # Create a new set of particles.
    num_particles = spl.alg.nparticles
    x = map(1:num_particles) do i
        if i != num_particles
            return AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG())
        else
            return reference
        end
    end
    particles = AdvancedPS.ParticleContainer(x, AdvancedPS.TracedRNG(), rng)

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl, reference)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    newreference = particles.vals[indx]

    # Compute the transition.
    _vi = newreference.model.f.varinfo
    transition = PGTransition(model, _vi, logevidence)

    return transition, PGState(_vi, newreference.rng)
end

function DynamicPPL.use_threadsafe_eval(
    ::DynamicPPL.SamplingContext{<:Sampler{<:Union{PG,SMC}}}, ::AbstractVarInfo
)
    return false
end

"""
    get_trace_local_varinfo_maybe(vi::AbstractVarInfo)

Get the `Trace` local varinfo if one exists.

If executed within a `TapedTask`, return the `varinfo` stored in the "taped globals" of the
task, otherwise return `vi`.
"""
function get_trace_local_varinfo_maybe(varinfo::AbstractVarInfo)
    trace = try
        Libtask.get_taped_globals(Any).other
    catch e
        e == KeyError(:task_variable) ? nothing : rethrow(e)
    end
    return (trace === nothing ? varinfo : trace.model.f.varinfo)::AbstractVarInfo
end

"""
    get_trace_local_varinfo_maybe(rng::Random.AbstractRNG)

Get the `Trace` local rng if one exists.

If executed within a `TapedTask`, return the `rng` stored in the "taped globals" of the
task, otherwise return `vi`.
"""
function get_trace_local_rng_maybe(rng::Random.AbstractRNG)
    return try
        Libtask.get_taped_globals(Any).rng
    catch e
        e == KeyError(:task_variable) ? rng : rethrow(e)
    end
end

"""
    set_trace_local_varinfo_maybe(vi::AbstractVarInfo)

Set the `Trace` local varinfo if executing within a `Trace`. Return `nothing`.

If executed within a `TapedTask`, set the `varinfo` stored in the "taped globals" of the
task. Otherwise do nothing.
"""
function set_trace_local_varinfo_maybe(vi::AbstractVarInfo)
    # TODO(mhauru) This should be done in a try-catch block, as in the commented out code.
    # However, Libtask currently can't handle this block.
    trace = #try
        Libtask.get_taped_globals(Any).other
    # catch e
    #     e == KeyError(:task_variable) ? nothing : rethrow(e)
    # end
    if trace !== nothing
        model = trace.model
        model = Accessors.@set model.f.varinfo = vi
        trace.model = model
    end
    return nothing
end

function DynamicPPL.assume(
    rng, ::Sampler{<:Union{PG,SMC}}, dist::Distribution, vn::VarName, vi::AbstractVarInfo
)
    arg_vi_id = objectid(vi)
    vi = get_trace_local_varinfo_maybe(vi)
    using_local_vi = objectid(vi) == arg_vi_id

    trng = get_trace_local_rng_maybe(rng)

    if ~haskey(vi, vn)
        r = rand(trng, dist)
        vi = push!!(vi, vn, r, dist)
    elseif DynamicPPL.is_flagged(vi, vn, "del")
        DynamicPPL.unset_flag!(vi, vn, "del") # Reference particle parent
        r = rand(trng, dist)
        vi[vn] = DynamicPPL.tovec(r)
        # TODO(mhauru):
        # The below is the only line that differs from assume called on SampleFromPrior.
        # Could we just call assume on SampleFromPrior and then `setorder!!` after that?
        vi = DynamicPPL.setorder!!(vi, vn, DynamicPPL.get_num_produce(vi))
    else
        r = vi[vn]
    end

    vi = DynamicPPL.accumulate_assume!!(vi, r, 0, vn, dist)

    # TODO(mhauru) Rather than this if-block, we should use try-catch within
    # `set_trace_local_varinfo_maybe`. However, currently Libtask can't handle such a block,
    # hence this.
    if !using_local_vi
        set_trace_local_varinfo_maybe(vi)
    end
    return r, vi
end

function DynamicPPL.tilde_observe!!(
    ctx::DynamicPPL.SamplingContext{<:Sampler{<:Union{PG,SMC}}}, right, left, vn, vi
)
    arg_vi_id = objectid(vi)
    vi = get_trace_local_varinfo_maybe(vi)
    using_local_vi = objectid(vi) == arg_vi_id

    left, vi = DynamicPPL.tilde_observe!!(ctx.context, right, left, vn, vi)

    # TODO(mhauru) Rather than this if-block, we should use try-catch within
    # `set_trace_local_varinfo_maybe`. However, currently Libtask can't handle such a block,
    # hence this.
    if !using_local_vi
        set_trace_local_varinfo_maybe(vi)
    end
    return left, vi
end

# Convenient constructor
function AdvancedPS.Trace(
    model::Model,
    sampler::Sampler{<:Union{SMC,PG}},
    varinfo::AbstractVarInfo,
    rng::AdvancedPS.TracedRNG,
)
    newvarinfo = deepcopy(varinfo)
    newvarinfo = DynamicPPL.reset_num_produce!!(newvarinfo)

    tmodel = TracedModel(model, sampler, newvarinfo, rng)
    newtrace = AdvancedPS.Trace(tmodel, rng)
    return newtrace
end

"""
    ProduceLogLikelihoodAccumulator{T<:Real} <: AbstractAccumulator

Exactly like `LogLikelihoodAccumulator`, but calls `Libtask.produce` on change of value.

# Fields
$(TYPEDFIELDS)
"""
struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.AbstractAccumulator
    "the scalar log likelihood value"
    logp::T
end

"""
    ProduceLogLikelihoodAccumulator{T}()

Create a new `ProduceLogLikelihoodAccumulator` accumulator with the log likelihood of zero.
"""
ProduceLogLikelihoodAccumulator{T}() where {T<:Real} =
    ProduceLogLikelihoodAccumulator(zero(T))
function ProduceLogLikelihoodAccumulator()
    return ProduceLogLikelihoodAccumulator{DynamicPPL.LogProbType}()
end

Base.copy(acc::ProduceLogLikelihoodAccumulator) = acc

function Base.show(io::IO, acc::ProduceLogLikelihoodAccumulator)
    return print(io, "ProduceLogLikelihoodAccumulator($(repr(acc.logp)))")
end

function Base.:(==)(
    acc1::ProduceLogLikelihoodAccumulator, acc2::ProduceLogLikelihoodAccumulator
)
    return acc1.logp == acc2.logp
end

function Base.isequal(
    acc1::ProduceLogLikelihoodAccumulator, acc2::ProduceLogLikelihoodAccumulator
)
    return isequal(acc1.logp, acc2.logp)
end

function Base.hash(acc::ProduceLogLikelihoodAccumulator, h::UInt)
    return hash((ProduceLogLikelihoodAccumulator, acc.logp), h)
end

# Note that this uses the same name as `LogLikelihoodAccumulator`. Thus only one of the two
# can be used in a given VarInfo.
DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood

function DynamicPPL.split(::ProduceLogLikelihoodAccumulator{T}) where {T}
    return ProduceLogLikelihoodAccumulator(zero(T))
end

function DynamicPPL.combine(
    acc::ProduceLogLikelihoodAccumulator, acc2::ProduceLogLikelihoodAccumulator
)
    return ProduceLogLikelihoodAccumulator(acc.logp + acc2.logp)
end

function Base.:+(
    acc1::ProduceLogLikelihoodAccumulator, acc2::DynamicPPL.LogLikelihoodAccumulator
)
    Libtask.produce(acc2.logp)
    return ProduceLogLikelihoodAccumulator(acc1.logp + acc2.logp)
end

function Base.zero(acc::ProduceLogLikelihoodAccumulator)
    return ProduceLogLikelihoodAccumulator(zero(acc.logp))
end

function DynamicPPL.accumulate_assume!!(
    acc::ProduceLogLikelihoodAccumulator, val, logjac, vn, right
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ProduceLogLikelihoodAccumulator, right, left, vn
)
    return acc +
           DynamicPPL.LogLikelihoodAccumulator(Distributions.loglikelihood(right, left))
end

function Base.convert(
    ::Type{ProduceLogLikelihoodAccumulator{T}}, acc::ProduceLogLikelihoodAccumulator
) where {T}
    return ProduceLogLikelihoodAccumulator(convert(T, acc.logp))
end
function DynamicPPL.convert_eltype(
    ::Type{T}, acc::ProduceLogLikelihoodAccumulator
) where {T}
    return ProduceLogLikelihoodAccumulator(convert(T, acc.logp))
end

# We need to tell Libtask which calls may have `produce` calls within them. In practice most
# of these won't be needed, because of inlining and the fact that `might_produce` is only
# called on `:invoke` expressions rather than `:call`s, but since those are implementation
# details of the compiler, we set a bunch of methods as might_produce = true. We start with
# adding to ProduceLogLikelihoodAccumulator, which is what calls `produce`, and go up the
# call stack.
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.accloglikelihood!!),Vararg}}) = true
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
function Libtask.might_produce(
    ::Type{<:Tuple{typeof(DynamicPPL.accumulate_observe!!),Vararg}}
)
    return true
end
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.tilde_observe!!),Vararg}}) = true
# Could the next two could have tighter type bounds on the arguments, namely a GibbsContext?
# That's the only thing that makes tilde_assume calls result in tilde_observe calls.
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.tilde_assume!!),Vararg}}) = true
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.tilde_assume),Vararg}}) = true
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.evaluate!!),Vararg}}) = true
function Libtask.might_produce(
    ::Type{<:Tuple{typeof(DynamicPPL.evaluate_threadsafe!!),Vararg}}
)
    return true
end
function Libtask.might_produce(
    ::Type{<:Tuple{typeof(DynamicPPL.evaluate_threadunsafe!!),Vararg}}
)
    return true
end
Libtask.might_produce(::Type{<:Tuple{<:DynamicPPL.Model,Vararg}}) = true
