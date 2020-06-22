#########################################
# Particle Transition (both SMC and PG) #
#########################################

"""
    ParticleTransition{T, F<:AbstractFloat}

Fields:
- `θ`: The parameters for any given sample.
- `lp`: The log pdf for the sample's parameters.
- `le`: The log evidence retrieved from the particle.
- `weight`: The weight of the particle the sample was retrieved from.
"""
struct ParticleTransition{T, F<:AbstractFloat}
    θ::T
    lp::F
    le::F
    weight::F
end

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le, :weight]
end

DynamicPPL.getlogp(t::ParticleTransition) = t.lp

############################
# Define a Sampler for SMC #
############################

"""
$(TYPEDEF)

Sequential Monte Carlo sampler.

# Fields

$(TYPEDFIELDS)
"""
struct SMC{space, R} <: ParticleInference
    resampler::R
end

"""
    SMC(space...)
    SMC([resampler = ResampleWithESSThreshold(), space = ()])
    SMC([resampler = resample_systematic, ]threshold[, space = ()])

Create a sequential Monte Carlo sampler of type [`SMC`](@ref) for the variables in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function SMC(resampler = Turing.Core.ResampleWithESSThreshold(), space::Tuple = ())
    return SMC{space, typeof(resampler)}(resampler)
end

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real, space::Tuple = ())
    return SMC(Turing.Core.ResampleWithESSThreshold(resampler, threshold), space)
end
SMC(threshold::Real, space::Tuple = ()) = SMC(resample_systematic, threshold, space)

# If only the space is defined
SMC(space::Symbol...) = SMC(space)
SMC(space::Tuple) = SMC(Turing.Core.ResampleWithESSThreshold(), space)

mutable struct SMCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
    particles            ::   ParticleContainer
end

function SMCState(model::Model)
    vi = VarInfo(model)
    particles = ParticleContainer(Trace[])

    return SMCState(vi, 0.0, particles)
end

function Sampler(alg::SMC, model::Model, s::Selector)
    dict = Dict{Symbol, Any}()
    state = SMCState(model)
    return Sampler(alg, dict, s, state)
end

################################################
# Overload the functions in mcmcsample for SMC #
################################################

function AbstractMCMC.sample_init!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    N::Integer;
    kwargs...
)
    # set the parameters to a starting value
    initialize_parameters!(spl; kwargs...)

    # reset the VarInfo
    vi = spl.state.vi
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)
    empty!(vi)

    # create a new set of particles
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    particles = T[Trace(model, spl, vi) for _ in 1:N]

    # create a new particle container
    spl.state.particles = pc = ParticleContainer(particles)

    # Perform particle sweep.
    logevidence = sweep!(pc, spl.alg.resampler)
    spl.state.average_logevidence = logevidence

    return
end

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    ::Integer,
    transition;
    iteration=-1,
    kwargs...
)
    # check that we received a real iteration number
    @assert iteration >= 1 "step! needs to be called with an 'iteration' keyword argument."

    # grab the weight
    pc = spl.state.particles
    weight = getweight(pc, iteration)

    # update the master vi
    particle = pc.vals[iteration]
    params = tonamedtuple(particle.vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(particle.vi)

    return ParticleTransition(params, lp, spl.state.average_logevidence, weight)
end

###########################
# Define a Sampler for PG #
###########################

"""
$(TYPEDEF)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

# Fields

$(TYPEDFIELDS)
"""
struct PG{space,R} <: ParticleInference
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

isgibbscomponent(::PG) = true

"""
    PG(n, space...)
    PG(n, [resampler = ResampleWithESSThreshold(), space = ()])
    PG(n, [resampler = resample_systematic, ]threshold[, space = ()])

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles for the variables
in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(
    nparticles::Int,
    resampler = Turing.Core.ResampleWithESSThreshold(),
    space::Tuple = (),
)
    return PG{space, typeof(resampler)}(nparticles, resampler)
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real, space::Tuple = ())
    return PG(nparticles, Turing.Core.ResampleWithESSThreshold(resampler, threshold), space)
end
function PG(nparticles::Int, threshold::Real, space::Tuple = ())
    return PG(nparticles, resample_systematic, threshold, space)
end

# If only the number of particles and the space is defined
PG(nparticles::Int, space::Symbol...) = PG(nparticles, space)
function PG(nparticles::Int, space::Tuple)
    return PG(nparticles, Turing.Core.ResampleWithESSThreshold(), space)
end

mutable struct PGState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
end

function PGState(model::Model)
    vi = VarInfo(model)
    return PGState(vi, 0.0)
end

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::PG, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = PGState(model)
    return Sampler(alg, info, s, state)
end


###############################################
# Overload the functions in mcmcsample for PG #
###############################################

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    ::Integer,
    transition;
    kwargs...
)
    # obtain or create reference particle
    vi = spl.state.vi
    ref_particle = isempty(vi) ? nothing : forkr(Trace(model, spl, vi))

    # reset the VarInfo before new sweep
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    # create a new set of particles
    num_particles = spl.alg.nparticles
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    if ref_particle === nothing
        particles = T[Trace(model, spl, vi) for _ in 1:num_particles]
    else
        particles = Vector{T}(undef, num_particles)
        @inbounds for i in 1:(num_particles - 1)
            particles[i] = Trace(model, spl, vi)
        end
        @inbounds particles[num_particles] = ref_particle
    end

    # create a new particle container
    pc = ParticleContainer(particles)

    # Perform a particle sweep.
    logevidence = sweep!(pc, spl.alg.resampler)

    # pick a particle to be retained.
    Ws = getweights(pc)
    indx = randcat(Ws)

    # extract the VarInfo from the retained particle.
    params = tonamedtuple(vi)
    spl.state.vi = pc.vals[indx].vi

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return ParticleTransition(params, lp, logevidence, 1.0)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::AbstractModel,
    spl::Sampler{<:ParticleInference},
    N::Integer,
    ts::Vector{<:ParticleTransition};
    resume_from = nothing,
    kwargs...
)
    # Exponentiate the average log evidence.
    # loge = exp(mean([t.le for t in ts]))
    loge = mean(t.le for t in ts)

    # If we already had a chain, grab the logevidence.
    if resume_from isa MCMCChains.Chains
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = resume_from.logevidence
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (pre_loge * pre_n + loge * N) / (pre_n + N)
    elseif resume_from !== nothing
        error("keyword argument `resume_from` has to be `nothing` or a `MCMCChains.Chains` object")
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
end

#################################################
# Overload assume and observe (both SMC and PG) #
#################################################

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:Union{PG,SMC}},
    dist::Distribution,
    vn::VarName,
    ::Any
)
    vi = current_trace().vi
    if inspace(vn, spl)
        if ~haskey(vi, vn)
            r = rand(rng, dist)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(rng, dist)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            updategid!(vi, vn, spl)
            r = vi[vn]
        end
    else # vn belongs to other sampler <=> conditionning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(rng, dist)
            push!(vi, vn, r, dist, Selector(:invalid))
        end
        lp = logpdf_with_trans(dist, r, istrans(vi, vn))
        acclogp!(vi, lp)
    end
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, value, vi)
    produce(logpdf(dist, value))
    return 0
end