### Attention this is a development package! It wount run.
### This file should not belong to this package.

###

### General SMC sampler with proposal distributions

### Conditional Particle Filter (Andireu  2010 https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)

### CPF-Ancestor Sampling (Lindsten 2014 http://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf)

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




####

#### Generic Sequential Monte Carlo sampler.

####



"""

    GSMC()



General Proposal Sequential Monte Carlo sampler.

This is an extension to the SMC sampler form the Turing repository.

It allows to set the proposal distributions.



Note that this method is particle-based, and arrays of variables

must be stored in a [`TArray`](@ref) object.



Fields:


- `resampler`: A function used to sample particles from the particle container.

  Defaults to `resample_systematic`.

- `resampler_threshold`: The threshold at which resampling terminates -- defaults to 0.5. If

  the `ess` <= `resampler_threshold` * `n_particles`, the resampling step is completed.



  Usage:



```julia

GSMC()


# Only specify for z a proposal distributon independent
# from the other parameters

GSMC(:x,:y,(:z , () -> f()))


# Only specify for z a proposal distributon
# which is dependent on x and y

GSMC(:x,:y,(:z,[:x,:y], (args) -> f(args)))

```

"""




struct GSMC{space, RT<:AbstractFloat} <: ParticleInference

    # Any contains a proposal distribution and a list of additional input variables

    proposals             ::  Dict{Symbol, Any}  # Proposals for paramters

    resampler             ::  Function

    resampler_threshold   ::  RT

end



alg_str(spl::Sampler{GSMC}) = "GSMC"


## Inspired from MH algorithm

function GSMC(
    resampler::Function,

    resampler_threshold::RT,

    space::Tuple

) where {RT<:AbstractFloat}

    new_space = ()

    proposals = Dict{Symbol,Any}()



    # parse random variables with their hypothetical proposal

    for element in space

        if isa(element, Symbol)

            new_space = (new_space..., element)

        else

            @assert isa(element[1], Symbol) "[GSMC] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, () -> Normal(0, 0.1)))"
            @assert length(element) == 2 || length(element) == 3 "[GSMC] ($element[1]) has wrong shape. Use syntax (:m, () -> Normal(0,0.1)) or (:z,[:x,:y], (args) -> f(args))"
            new_space = (new_space..., element[1])
            if length(element)== 2
                proposals[element[1]] = (Vector{Symbol}(undef,0), element[2]) #No input arguments
            elseif length(element) == 3
                @assert isa(element[2],Vector{Symbol}) "[GSMC] For length three elements, ($element[2]) should be a Vecotr{Symbol}"
                proposal[element[1]] = (element[2], element[3])
            end
        end

    end

    return GSMC{new_space, RT}(proposals ,resampler, resampler_threshold)

end

GSMC() = GSMC(resample_systematic, 0.5, ())

GSMC(::Tuple{}) = GSMC()

function SMC(space::Symbol...)

    GSMC(resample_systematic, 0.5, space)

end


## We can use the SMCState from the Turing repo

# mutable struct SMCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
#
#     vi                   ::   V
#
#     # The logevidence after aggregating all samples together.
#
#     average_logevidence  ::   F
#
#     particles            ::   ParticleContainer
#
# end
#
#
#
# function SMCState(
#
#     model::M,
#
# ) where {
#
#     M<:Model
#
# }
#
#     vi = VarInfo(model)
#
#     particles = ParticleContainer{Trace}(model)
#
#
#
#     return SMCState(vi, 0.0, particles)
#
# end

## We leave this function unchanged

function Sampler(alg::T, model::Model, s::Selector) where T<:GSMC

    dict = Dict{Symbol, Any}()

    state = SMCState(model)

    return Sampler(alg, dict, s, state)

end

## We leave this function unchanged

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

    spl.state.particles = ParticleContainer{Trace{typeof(spl),

        typeof(spl.state.vi), typeof(model)}}(model)



    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.

    set_retained_vns_del_by_spl!(spl.state.vi, spl)

    resetlogp!(spl.state.vi)



    push!(spl.state.particles, N, spl, empty!(spl.state.vi))



    while consume(spl.state.particles) != Val{:done}

        ess = effectiveSampleSize(spl.state.particles)

        if ess <= spl.alg.resampler_threshold * length(spl.state.particles)

            resample!(spl.state.particles, spl.alg.resampler)

        end

    end

end

## We leave this function unchanged

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

    Ws = weights(spl.state.particles)



    # update the master vi.

    particle = spl.state.particles.vals[iteration]

    params = tonamedtuple(particle.vi)

    lp = getlogp(particle.vi)



    return ParticleTransition(params, lp, spl.state.particles.logE, Ws[iteration])

end



####

#### Particle Gibbs sampler.

####



"""

    CPF(n_particles::Int,space)



Conditional Particle Filter sampler.



Note that this method is particle-based, and arrays of variables

must be stored in a [`TArray`](@ref) object.



Usage:



```julia

CPF(100)

```

"""

struct CPF{space} <: ParticleInference


  proposals             ::    Dict{Symbol,Any}()  # Proposal distributions

  n_particles           ::    Int         # number of particles used

  resampler             ::    Function    # function to resample

end

function CPF(n_particles::Int, resampler::Function, space::Tuple)

    new_space = ()

    proposals = Dict{Symbol,Any}()



    # parse random variables with their hypothetical proposal

    for element in space

        if isa(element, Symbol)

            new_space = (new_space..., element)

        else

            @assert isa(element[1], Symbol) "[GSMC] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, () -> Normal(0, 0.1)))"
            @assert length(element) == 2 || length(element) == 3 "[GSMC] ($element[1]) has wrong shape. Use syntax (:m, () -> Normal(0,0.1)) or (:z,[:x,:y], (args) -> f(args))"
            new_space = (new_space..., element[1])
            if length(element)== 2
                proposals[element[1]] = (Vector{Symbol}(undef,0), element[2]) #No input arguments
            elseif length(element) == 3
                @assert isa(element[2],Vector{Symbol}) "[GSMC] For length three elements, ($element[2]) should be a Vecotr{Symbol}"
                proposal[element[1]] = (element[2], element[3])
            end
        end

    end

    return GSMC{new_space, RT}(proposals ,n_particles,resampler)
end

CPF(n1::Int, ::Tuple{}) = CPF(n1)

function CPF(n1::Int, space::Symbol...)

    CPF(n1, resample_systematic, space)

end



alg_str(spl::Sampler{CPF}) = "CPF"

## We resue this...
#
# mutable struct PGState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
#
#     vi                   ::   V
#
#     # The logevidence after aggregating all samples together.
#
#     average_logevidence  ::   F
#
# end
#
#
#
# function PGState(model::M) where {M<:Model}
#
#     vi = VarInfo(model)
#
#     return PGState(vi, 0.0)
#
# end






"""

    Sampler(alg::PG, model::Model, s::Selector)



Return a `Sampler` object for the PG algorithm.

"""

function Sampler(alg::T, model::Model, s::Selector) where T<:CPF

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

    particles = ParticleContainer{Trace{typeof(spl), typeof(spl.state.vi), typeof(model)}}(model)



    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep.

    ref_particle = isempty(spl.state.vi) ?

              nothing :

              forkr(Trace(model, spl, spl.state.vi))



    set_retained_vns_del_by_spl!(spl.state.vi, spl)

    resetlogp!(spl.state.vi)



    if ref_particle === nothing

        push!(particles, spl.alg.n_particles, spl, spl.state.vi)

    else

        push!(particles, spl.alg.n_particles-1, spl, spl.state.vi)

        push!(particles, ref_particle)

    end



    while consume(particles) != Val{:done}

        resample!(particles, spl.alg.resampler, ref_particle)

    end



    ## pick a particle to be retained.

    Ws = weights(particles)

    indx = randcat(Ws)



    # Extract the VarInfo from the retained particle.

    params = tonamedtuple(spl.state.vi)

    spl.state.vi = particles[indx].vi

    lp = getlogp(spl.state.vi)



    # update the master vi.

    return ParticleTransition(params, lp, particles.logE, 1.0)

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

## We need to sample form the proposal instead of the transition

function assume(  spl::Sampler{T},

                  dist::Distribution,

                  vn::VarName,

                  _::VarInfo

                ) where T<:Union{CPF,GSMC}


    ## The part concerning the tasks is left unchanged !

    vi = current_trace().vi

    if isempty(getspace(spl.alg)) || vn.sym in getspace(spl.alg)

        if ~haskey(vi, vn)

            ## Here, our changings start...
            ## Attention, we do not check for the support of the proposal distribution!!
            ## We implicitely assume so far that the support is not violated!!
            ## We changed r = rand(dist) with the following code in order to include proposal distributions.

            if vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter


                tuple = spl.alg.proposals[vn.sym]()

                #First extract the argument variables
                extracted_symbosl = tuple[1]
                if isempty(extracted_symbosl)
                    proposal = tuple[2]()
                else

                    args = []
                    for sym in extracted_symbosl
                        if isempty(getspace(spl.alg)) || sym in getspace(spl.alg)
                            error("[CPF] Symbol ($sym) is not defined yet. The arguments for the propsal must occur before!")
                        end
                        push!(args,vi[sym])
                    end
                    proposal = tuble[2](args)
                end
                r = rand(proposal)
                ## In this case, we need to set the logp because we do not sample directly from the prior!
                acclogp!(vi, logpdf(dist, r) -logpdf(proposal, r))

                end
            else # Prior as proposal

                r = rand(dist)

                spl.state.prior_prob += logpdf(dist, r) # accumulate prior for PMMH

            end


            push!(vi, vn, r, dist, spl)

        elseif is_flagged(vi, vn, "del")



            unset_flag!(vi, vn, "del")


            ## Here, our changings start...
            ## Attention, we do not check for the support of the proposal distribution!!
            ## We implicitely assume so far that the support is not! violated

            if vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter


                tuple = spl.alg.proposals[vn.sym]()

                #First extract the argument variables
                extracted_symbosl = tuple[1]
                if isempty(extracted_symbosl)
                    proposal = tuple[2]()
                else

                    args = []
                    for sym in extracted_symbosl
                        if isempty(getspace(spl.alg)) || sym in getspace(spl.alg)
                            error("[CPF] Symbol ($sym) is not defined yet. The arguments for the propsal must occur before!")
                        end
                        push!(args,vi[sym])
                    end
                    proposal = tuble[2](args)
                end
                r = rand(proposal)

                ## In this case, we need to set the logp because we do not sample directly from the prior!
                acclogp!(vi, logpdf(dist, r) -logpdf(proposal, r))

                end
            else # Prior as proposal

                r = rand(dist)

                spl.state.prior_prob += logpdf(dist, r) # accumulate prior for PMMH

            end

            vi[vn] = vectorize(dist, r)

            setgid!(vi, spl.selector, vn)

            setorder!(vi, vn, vi.num_produce)

        else

            updategid!(vi, vn, spl)

            r = vi[vn]

        end

    else # vn belongs to other sampler <=> conditionning on vn

        if haskey(vi, vn)

            r = vi[vn]

        else # What happens here? I assume that vn does not belong to any sampler - how is this possible
             # Should the sanity check not prevent his
            r = rand(dist)

            push!(vi, vn, r, dist, Selector(:invalid))

        end

        acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    end

    return r, zero(Real)

end

## Only need to change A<:Union{CPF,GSMC}

function assume(  spl::Sampler{A},

                  dists::Vector{D},

                  vn::VarName,

                  var::Any,

                  vi::VarInfo

                ) where {A<:Union{PG,SMC},D<:Distribution}

    error("[Turing] CPF and GSMC doesn't support vectorizing assume statement")

end

## Only need to change A<:Union{CPF,GSMC}


function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:Union{CPF,GSMC}

    produce(logpdf(dist, value))

    return zero(Real)

end


## Only need to change A<:Union{CPF,GSMC}

function observe( spl::Sampler{A},

                  ds::Vector{D},

                  value::Any,

                  vi::VarInfo

                ) where {A<:Union{CPF,GSMC},D<:Distribution}

    error("[Turing] CPF and GSMC doesn't support vectorizing observe statement")

end



for alg in (:GSMC,:CPF)

    @eval getspace(::$alg{space}) where {space} = space

    @eval getspace(::Type{<:$alg{space}}) where {space} = space

###########################################################################################
## We do not change these - therefore, we use the code directly from the turing repo
#
#
# ####
#
# #### Resampling schemes for particle filters
#
# ####
#
#
#
# # Some references
#
# #  - http://arxiv.org/pdf/1301.4019.pdf
#
# #  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
#
# # Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering
#
#
#
# # Default resampling scheme
#
# function resample(w::AbstractVector{<:Real}, num_particles::Integer=length(w))
#
#     return resample_systematic(w, num_particles)
#
# end
#
#
#
# # More stable, faster version of rand(Categorical)
#
# function randcat(p::AbstractVector{T}) where T<:Real
#
#     r, s = rand(T), 1
#
#     for j in eachindex(p)
#
#         r -= p[j]
#
#         if r <= zero(T)
#
#             s = j
#
#             break
#
#         end
#
#     end
#
#     return s
#
# end
#
#
#
# function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
#
#     return rand(Distributions.sampler(Categorical(w)), num_particles)
#
# end
#
#
#
# function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer)
#
#
#
#     M = length(w)
#
#
#
#     # "Repetition counts" (plus the random part, later on):
#
#     Ns = floor.(length(w) .* w)
#
#
#
#     # The "remainder" or "residual" count:
#
#     R = Int(sum(Ns))
#
#
#
#     # The number of particles which will be drawn stocastically:
#
#     M_rdn = num_particles - R
#
#
#
#     # The modified weights:
#
#     Ws = (M .* w - floor.(M .* w)) / M_rdn
#
#
#
#     # Draw the deterministic part:
#
#     indx1, i = Array{Int}(undef, R), 1
#
#     for j in 1:M
#
#         for k in 1:Ns[j]
#
#             indx1[i] = j
#
#             i += 1
#
#         end
#
#     end
#
#
#
#     # And now draw the stocastic (Multinomial) part:
#
#     return append!(indx1, rand(Distributions.sampler(Categorical(w)), M_rdn))
#
# end
#
#
#
# function resample_stratified(w::AbstractVector{<:Real}, num_particles::Integer)
#
#
#
#     Q, N = cumsum(w), num_particles
#
#
#
#     T = Array{Float64}(undef, N + 1)
#
#     for i=1:N,
#
#         T[i] = rand() / N + (i - 1) / N
#
#     end
#
#     T[N+1] = 1
#
#
#
#     indx, i, j = Array{Int}(undef, N), 1, 1
#
#     while i <= N
#
#         if T[i] < Q[j]
#
#             indx[i] = j
#
#             i += 1
#
#         else
#
#             j += 1
#
#         end
#
#     end
#
#     return indx
#
# end
#
#
#
# function resample_systematic(w::AbstractVector{<:Real}, num_particles::Integer)
#
#
#
#     Q, N = cumsum(w), num_particles
#
#
#
#     T = collect(range(0, stop = maximum(Q)-1/N, length = N)) .+ rand()/N
#
#     push!(T, 1)
#
#
#
#     indx, i, j = Array{Int}(undef, N), 1, 1
#
#     while i <= N
#
#         if T[i] < Q[j]
#
#             indx[i] = j
#
#             i += 1
#
#         else
#
#             j += 1
#
#         end
#
#     end
#
#     return indx
#
# end
