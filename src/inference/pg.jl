
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
    PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold(), space = ()])
    PG(n, [resampler = AdvancedPS.resample_systematic, ]threshold[, space = ()])

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles for the variables
in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(
    nparticles::Int,
    resampler = AdvancedPS.ResampleWithESSThreshold(),
    space::Tuple = (),
)
    return PG{space, typeof(resampler)}(nparticles, resampler)
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real, space::Tuple = ())
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(resampler, threshold), space)
end
function PG(nparticles::Int, threshold::Real, space::Tuple = ())
    return PG(nparticles, AdvancedPS.resample_systematic, threshold, space)
end

# If only the number of particles and the space is defined
PG(nparticles::Int, space::Symbol...) = PG(nparticles, space)
function PG(nparticles::Int, space::Tuple)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(), space)
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
    ref_particle = isempty(vi) ? nothing : AdvancedPS.forkr(AdvancedPS.Trace(model, spl, vi))

    # reset the VarInfo before new sweep
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    # create a new set of particles
    num_particles = spl.alg.nparticles
    T = AdvancedPS.Trace{typeof(spl),typeof(vi),typeof(model)}
    if ref_particle === nothing
        particles = T[AdvancedPS.Trace(model, spl, vi) for _ in 1:num_particles]
    else
        particles = Vector{T}(undef, num_particles)
        @inbounds for i in 1:(num_particles - 1)
            particles[i] = AdvancedPS.Trace(model, spl, vi)
        end
        @inbounds particles[num_particles] = ref_particle
    end

    # create a new particle container
    pc = AdvancedPS.ParticleContainer(particles)

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(pc, spl.alg.resampler)

    # pick a particle to be retained.
    Ws = AdvancedPS.getweights(pc)
    indx = AdvancedPS.randcat(Ws)

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

###############################
# Overload assume and observe #
###############################

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:PG},
    dist::Distribution,
    vn::VarName,
    ::Any
)
    vi = AdvancedPS.current_trace().vi
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

function DynamicPPL.observe(spl::Sampler{<:PG}, dist::Distribution, value, vi)
    produce(logpdf(dist, value))
    return 0
end