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
    SMC([resampler = AdvancedPS.ResampleWithESSThreshold(), space = ()])
    SMC([resampler = AdvancedPS.resample_systematic, ]threshold[, space = ()])

Create a sequential Monte Carlo sampler of type [`SMC`](@ref) for the variables in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function SMC(resampler = AdvancedPS.ResampleWithESSThreshold(), space::Tuple = ())
    return SMC{space, typeof(resampler)}(resampler)
end

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real, space::Tuple = ())
    return SMC(AdvancedPS.ResampleWithESSThreshold(resampler, threshold), space)
end
SMC(threshold::Real, space::Tuple = ()) = SMC(AdvancedPS.resample_systematic, threshold, space)

# If only the space is defined
SMC(space::Symbol...) = SMC(space)
SMC(space::Tuple) = SMC(AdvancedPS.ResampleWithESSThreshold(), space)

mutable struct SMCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
    particles            ::   AdvancedPS.ParticleContainer
end

function SMCState(model::Model)
    vi = VarInfo(model)
    particles = AdvancedPS.ParticleContainer(AdvancedPS.Trace[])

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
    T = AdvancedPS.Trace{typeof(spl),typeof(vi),typeof(model)}
    particles = T[AdvancedPS.Trace(model, spl, vi) for _ in 1:N]

    # create a new particle container
    spl.state.particles = pc = AdvancedPS.ParticleContainer(particles)

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(pc, spl.alg.resampler)
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
    weight = AdvancedPS.getweight(pc, iteration)

    # update the master vi
    particle = pc.vals[iteration]
    params = tonamedtuple(particle.vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(particle.vi)

    return ParticleTransition(params, lp, spl.state.average_logevidence, weight)
end

###############################
# Overload assume and observe #
###############################

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:SMC},
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

function DynamicPPL.observe(spl::Sampler{<:SMC}, dist::Distribution, value, vi)
    produce(logpdf(dist, value))
    return 0
end