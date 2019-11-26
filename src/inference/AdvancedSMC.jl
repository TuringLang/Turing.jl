###
### Particle Filtering and Particle MCMC Samplers.
###

#######################
# Particle Transition #
#######################

"""
    ParticleTransition{T, F<:AbstractFloat} <: AbstractTransition

Fields:
- `θ`: The parameters for any given sample.
- `lp`: The log pdf for the sample's parameters.
- `le`: The log evidence retrieved from the particle.
- `weight`: The weight of the particle the sample was retrieved from.
"""
struct ParticleTransition{T, F<:AbstractFloat} <: AbstractTransition
    θ::T
    lp::F
    le::F
    weight::F
end

transition_type(spl::Sampler{<:ParticleInference}) = ParticleTransition

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp, :le, :weight]
end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC()

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Fields:
- `resampler`: A function used to sample particles from the particle container.
  Defaults to `resample_systematic`.
- `resampler_threshold`: The threshold at which resampling terminates -- defaults to 0.5. If
  the `ess` <= `resampler_threshold` * `n_particles`, the resampling step is completed.

  Usage:

```julia
SMC()
```
"""
struct SMC{space, RT<:AbstractFloat} <: ParticleInference
    resampler             ::  Function
    resampler_threshold   ::  RT
    ps_alg!                ::  Function

end

alg_str(spl::Sampler{SMC}) = "SMC"
function SMC(
    resampler::Function,
    resampler_threshold::RT,
    space::Tuple
) where {RT<:AbstractFloat}
    ps_alg = APS.sampleSMC!
    return SMC{space, RT}(resampler, resampler_threshold, ps_alg)
end
SMC() = SMC(APS.resample_systematic, 0.5, ())
SMC(::Tuple{}) = SMC()
function SMC(space::Symbol...)
    SMC(APS.resample_systematic, 0.5, space)
end

mutable struct SMCState{V<:VarInfo, F<:AbstractFloat,T} <: AbstractSamplerState where T<:APS.ParticleContainer{typeof(V),APS.PGTaskInfo}
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
    particles            ::   T

end

function SMCState(
    model::M,
) where {
    M<:Model
}
    vi = VarInfo(model)
    particles = APS.ParticleContainer{typeof(vi),APS.PGTaskInfo}()

    return SMCState{typeof(vi),Float64,typeof(particles)}(vi, 0.0, particles)
end

function Sampler(alg::T, model::Model, s::Selector) where T<:SMC
    dict = Dict{Symbol, Any}()
    state = SMCState(model)
    return Sampler(alg, dict, s, state)
end

function sample_init!(
    ::AbstractRNG,
    model::Turing.Model,
    spl::Sampler{<:SMC},
    N::Integer;
    kwargs...
)
    # Set the parameters to a starting value.
    initialize_parameters!(spl; kwargs...)

    # Update the particle container now that the sampler type
    # is defined.
    spl.state.particles.manipulators["set_retained_vns_del_by_spl!"] = get_srvndbs(spl)
    spl.state.particles.manipulators["copy"] = Turing.deepcopy

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    task = CTask( () -> begin vi_new=model(spl.state.vi,spl); produce(Val{:done}); vi_new; end )
    taskinfo = APS.PGTaskInfo(0.0,0.0)
    APS.extend!(spl.state.particles, N, empty!(spl.state.vi), task , taskinfo)
    spl.alg.ps_alg!(spl.state.particles,spl.alg.resampler,spl.alg.resampler_threshold)


end


function step!(
    ::AbstractRNG,
    model::Turing.Model,
    spl::Sampler{<:SMC},
    ::Integer;
    iteration=-1,
    kwargs...
)
    # Check that we received a real iteration number.
    @assert iteration >= 1 "step! needs to be called with an 'iteration' keyword argument."

    ## Grab the weights.
    Ws = APS.weights(spl.state.particles)

    # update the master vi.
    particle = spl.state.particles[iteration]
    params = tonamedtuple(particle.vi)


    return ParticleTransition(params, particle.taskinfo.logp, spl.state.particles.logE, Ws[iteration])
end

####
#### Particle Gibbs sampler.
####

"""
    PG(n_particles::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
```
"""
struct PG{space} <: ParticleInference
  n_particles           ::    Int         # number of particles used
  resampler             ::    Function    # function to resample
  ps_alg!                ::    Function    # particle sampling algorithm
end
function PG(n_particles::Int, resampler::Function, space::Tuple)
    ps_alg! = APS.samplePG!
    return PG{space}(n_particles, resampler,ps_alg!)
end
PG(n1::Int, ::Tuple{}) = PG(n1)
function PG(n1::Int, space::Symbol...)
    PG(n1, APS.resample_systematic, space)
end

alg_str(spl::Sampler{PG}) = "PG"

mutable struct PGState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
end

function PGState(model::M) where {M<:Model}
    vi = VarInfo(model)
    return PGState(vi, 0.0)
end

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::T, model::Model, s::Selector) where T<:PG
    info = Dict{Symbol, Any}()
    state = PGState(model)
    return Sampler(alg, info, s, state)
end

function step!(
    ::AbstractRNG,
    model::Turing.Model,
    spl::Sampler{<:PG},
    ::Integer;
    kwargs...
)
    vi = spl.state.vi
    task = CTask( () -> begin vi_new=model(vi,spl); produce(Val{:done}); vi_new; end )
    taskinfo = APS.PGTaskInfo(0.0, 0.0)
    particles = APS.ParticleContainer{typeof(vi), typeof(taskinfo)}()

    particles.manipulators["set_retained_vns_del_by_spl!"] = get_srvndbs(spl)
    particles.manipulators["copy"] = Turing.deepcopy

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep.

    # This is very inefficient because the state will be copied twice!.
    # However, this is not a problem to change.
    ref_particle = isempty(spl.state.vi) ?
              nothing :
              deepcopy(spl.state.vi)


    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    if ref_particle === nothing
        APS.extend!(particles, spl.alg.n_particles, spl.state.vi, task , taskinfo)
        spl.alg.ps_alg!(particles,spl.alg.resampler)

    else
        APS.extend!(particles, spl.alg.n_particles-1, spl.state.vi, task , taskinfo)
        APS.extend!(particles, 1, ref_particle, task, taskinfo)
        spl.alg.ps_alg!(particles,spl.alg.resampler, particles[spl.alg.n_particles])
    end

    ## pick a particle to be retained.
    Ws = APS.weights(particles)
    indx = APS.randcat(Ws)

    # Extract the VarInfo from the retained particle.
    params = tonamedtuple(spl.state.vi)
    spl.state.vi = particles[indx].vi

    ## This is kind of weired... what excalty do we want?
    # Original : lp = getlogp(spl.state.vi)
    # update the master vi.
    return ParticleTransition(params, particles[indx].taskinfo.logp, particles.logE, 1.0)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ParticleInference},
    N::Integer,
    ts::Vector{ParticleTransition};
    kwargs...
)
    # Set the default for resuming the sampler.
    resume_from = get(kwargs, :resume_from, nothing)

    # Exponentiate the average log evidence.
    # loge = exp(mean([t.le for t in ts]))
    loge = mean(t.le for t in ts)

    # If we already had a chain, grab the logevidence.
    if resume_from !== nothing   # concat samples
        @assert resume_from isa Chains "resume_from needs to be a Chains object."
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = resume_from.logevidence
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (pre_loge * pre_n + loge * N) / (pre_n + N)
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
end




####
#### Particle Gibbs with Ancestor Sampling sampler.
####

"""
    PGAS(n_particles::Int, joint_logp::Function)

Particle Gibbs with Ancestor Sampling sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Fields:
- `joint_logp`: A function which takes a Dict{Symbol,Any} as input
and returns the log pdf corresponding to log p(x_{0:T}) from the
original paper. The Symobls of the dict must be contained in vi!


Usage:

```julia
PG(100, 100)
```
"""
struct PGAS{space} <: ParticleInference
  n_particles           ::    Int         # number of particles used
  resampler             ::    Function    # function to resample
  ps_alg!               ::    Function    # particle sampling algorithm
  joint_logp            ::    Union{Function,Nothing} #Joint log pdf function of the transistions
end
function PGAS(n_particles::Int, resampler::Function, joint_logp::Union{Function,Nothing}, space::Tuple)
    ps_alg! = APS.samplePGAS!
    return PGAS{space}(n_particles, resampler,ps_alg!,joint_logp)
end
PGAS(n1::Int, ::Tuple{}) = PGAS(n1)
function PGAS(n1::Int, space::Symbol...)
    PGAS(n1, APS.resample_systematic, nothing,   space)
end
function PGAS(n1::Int, joint_logp::Union{Function,Nothing}, space::Symbol...)
    PGAS(n1, APS.resample_systematic, joint_logp,   space)
end

alg_str(spl::Sampler{PGAS}) = "PGAS"

mutable struct PGASState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
end

function PGASState(model::M) where {M<:Model}
    vi = VarInfo(model)
    return PGASState(vi, 0.0)
end


"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::T, model::Model, s::Selector) where T<:PGAS
    info = Dict{Symbol, Any}()
    state = PGASState(model)
    return Sampler(alg, info, s, state)
end

function step!(
    ::AbstractRNG,
    model::Turing.Model,
    spl::Sampler{<:PGAS},
    ::Integer;
    kwargs...
)
    vi = spl.state.vi
    task = CTask( () -> begin vi_new=model(vi,spl); produce(Val{:done}); vi_new; end )

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep.

    # This is very inefficient because the state will be copied twice!.
    # However, this is not a problem to change.
    ref_particle = isempty(spl.state.vi) ?
              nothing :
              deepcopy(spl.state.vi)
    if ref_particle === nothing
        taskinfo = APS.PGASTaskInfo(0.0, 0.0,false)
    else
        taskinfo = APS.PGASTaskInfo(0.0, 0.0,true)
    end

    particles = APS.ParticleContainer{typeof(vi), typeof(taskinfo)}()

    particles.manipulators["set_retained_vns_del_by_spl!"] = get_srvndbs(spl)
    particles.manipulators["copy"] = Turing.deepcopy
    particles.manipulators["merge_traj"] = generate_merge_traj(spl)




    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    if ref_particle === nothing
        APS.extend!(particles, spl.alg.n_particles, spl.state.vi, task , taskinfo)
        spl.alg.ps_alg!(particles,spl.alg.resampler)
    else
        APS.extend!(particles, spl.alg.n_particles-1, spl.state.vi, task , taskinfo)
        APS.extend!(particles, 1, ref_particle, task, taskinfo)
        spl.alg.ps_alg!(particles,spl.alg.resampler, particles[spl.alg.n_particles],spl.alg.joint_logp)
    end

    ## pick a particle to be retained.
    Ws = APS.weights(particles)
    indx = APS.randcat(Ws)

    # Extract the VarInfo from the retained particle.
    params = tonamedtuple(spl.state.vi)
    spl.state.vi = particles[indx].vi

    ## This is kind of weired... what excalty do we want?
    # Original : lp = getlogp(spl.state.vi)
    # update the master vi.
    return ParticleTransition(params, particles[indx].taskinfo.logp, particles.logE, 1.0)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ParticleInference},
    N::Integer,
    ts::Vector{ParticleTransition};
    kwargs...
)
    # Set the default for resuming the sampler.
    resume_from = get(kwargs, :resume_from, nothing)

    # Exponentiate the average log evidence.
    # loge = exp(mean([t.le for t in ts]))
    loge = mean(t.le for t in ts)

    # If we already had a chain, grab the logevidence.
    if resume_from !== nothing   # concat samples
        @assert resume_from isa Chains "resume_from needs to be a Chains object."
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = resume_from.logevidence
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (pre_loge * pre_n + loge * N) / (pre_n + N)
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
end





function assume(  spl::Sampler{T},
                  dist::Distribution,
                  vn::VarName,
                  _::VarInfo
                ) where T<:Union{PG,SMC,PGAS}

    vi = APS.current_trace().vi
    taskinfo = APS.current_trace().taskinfo
    if isempty(getspace(spl.alg)) || vn.sym in getspace(spl.alg)
        if ~haskey(vi, vn)



            r = rand(dist)



            taskinfo.logpseq += logpdf(dist,r)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")


            r = rand(dist)

            taskinfo.logpseq += logpdf(dist,r)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, vi.num_produce)
        else
            # Ancestor sampling will make use of this.
            updategid!(vi, vn, spl)
            r = vi[vn]
            taskinfo.logpseq += logpdf(dist,r)
        end
    else # vn belongs to other sampler <=> conditionning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(dist)
            push!(vi, vn, r, dist, Selector(:invalid))
        end
        taskinfo.logp += logpdf_with_trans(dist, r, istrans(vi, vn))
    end
    return r, zero(Real)
end




function assume(  spl::Sampler{A},
                  dists::Vector{D},
                  vn::VarName,
                  var::Any,
                  vi::VarInfo
                ) where {A<:Union{PG,SMC,PGAS},D<:Distribution}
    error("[Turing] PG and SMC don't support vectorizing assume statement")
end

function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:Union{PG,SMC}
    produce(logpdf(dist, value))
    return zero(Real)
end
function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:PGAS
    produce(logpdf(dist, value))

    # In order to prevent an excessive amount of copying, we need to have a high level of synchorinty in ancestor sampling...
    if APS.current_trace().taskinfo.hold
        produce(0.0)
    end

    return zero(Real)
end




function observe( spl::Sampler{A},
                  ds::Vector{D},
                  value::Any,
                  vi::VarInfo
                ) where {A<:Union{PGAS,PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing observe statement")
end


function get_srvndbs(spl::Sampler{A}) where A<:Union{PG,SMC,PGAS}
    function srvdbs(x::VarInfo)
        set_retained_vns_del_by_spl!(x,spl)
    end
end

#This is used for ancestor sampling. We simply merge the two trajectories.
function generate_merge_traj(spl::Sampler{A}) where A<:Union{PG,SMC,PGAS}
    ## Extends the vi trajectory by the ref_traj!
    function merge_traj!(vi::VarInfo, ref_traj::VarInfo,num_produce::Int64=-1)
        # We go trough all the variables in the space
        for vn in _getvns(ref_traj,spl)
            # Please check this !!!
            if num_produce == -1 || getorder(ref_traj,vn[1]) <= num_produce
                # And copy all the missing variables.
                if ~haskey(vi, vn[1])
                    push!(vi, vn, ref_traj[vn], getdist(ref_traj,vn), spl)
                elseif is_flagged(vi, vn[1], "del")
                    unset_flag!(vi, vn[1], "del")
                    vi[vn[1]] = ref_traj[vn]
                    setgid!(vi, spl.selector, vn[1])
                    setorder!(vi, vn[1], vi.num_produce)
                end
            end
        end
        vi
    end
end
