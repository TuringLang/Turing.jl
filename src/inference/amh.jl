import AdvancedMH
const AMH = AdvancedMH

import DynamicPPL: VarName, _getranges, _getindex, getval, _getvns

###
### Sampler states
###

struct MH{space} <: InferenceAlgorithm end

MH(space...) = MH{space}()

mutable struct MHState{VI<:VarInfo, M<:AMH.Metropolis} <: AbstractSamplerState
    vi :: VI
    mh_sampler :: M
    density_model :: AMH.DensityModel
    transitions :: Dict{VarName, AMH.Transition}
end

alg_str(::Sampler{<:MH}) = "MH"

#################
# MH Transition #
#################

struct MHTransition{T, F<:AbstractFloat, M<:AMH.Transition} <: AbstractTransition
    θ    :: T
    lp   :: F
    mh_trans :: M
end

function MHTransition(spl::Sampler{<:Union{MH, RWMH}}, mh_trans::AMH.Transition)
    theta = tonamedtuple(spl.state.vi)
    return MHTransition(theta, mh_trans.lp, mh_trans)
end

transition_type(spl::Sampler{<:MH}) = MHTransition
    # typeof(MHTransition(spl, AMH.Transition(spl.state.density_model, spl.state.vi[spl])))

additional_parameters(::Type{<:MHTransition}) = [:lp]

function gen_logπ_mh(vi::VarInfo, spl::Sampler, model)
    function logπ(x)::Float64
        x_old, lj_old = vi[spl], vi.logp
        # vi[spl] = [x]
        set_vi_vals!(vi, x)
        runmodel!(model, vi, spl)
        lj = vi.logp
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

function set_vi_vals!(vi::VarInfo, nt::NamedTuple{names}) where names
    for name in names
        vns = vi.metadata[name].vns
        vals = nt[name]
        if vals isa Real
            vi[vns[1]] = [vals]
        else
            for vn in vns
                vi[vn] = vals
            end
        end
    end
end

###############################
# Static MH (from prior only) #
###############################

function Sampler(
    alg::MH,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Set up VarInfo.
    vi = VarInfo(model)

    # Create a vector of prior distributions.
    space = getspace(alg)
    vns = _getvns(vi, s, Val(space))

    # Generate a sampler to retrieve the initial theta.
    spl = Sampler(alg, info, s, SamplerState(vi))

    # Retrieve all the proposal distributions and initial parameters.
    dists = Distribution[]
    init_theta = []

    for (i, (key, vn)) in enumerate(pairs(vns))
        for dist in vi.metadata[key].dists
            push!(dists, dist)
        end
        push!(init_theta, first(getindex.(Ref(vi), vi.metadata[key].vns)))
    end

    # Create a NamedTuple of intial params and distributions.
    syms = collect(keys(vns))
    init_theta_nt = NamedTuple{tuple(syms...)}(tuple(init_theta...))
    dists_nt = NamedTuple{tuple(syms...)}(tuple(dists...))

    # Make a sampler state, using a dummy density model.
    state = MHState(vi, AMH.StaticMH(init_theta_nt, dists_nt), AMH.DensityModel(x -> 0.0), Dict{VarName, AMH.Transition}())

    # Generate a sampler.
    spl = Sampler(alg, info, s, state)

    # Create the actual densitymodel.
    spl.state.density_model = AMH.DensityModel(gen_logπ_mh(vi, spl, model))

    return spl
end

##################
# Random walk MH #
##################

struct RWMH{space} <: InferenceAlgorithm end
RWMH(space...) = RWMH{space}()
alg_str(::Sampler{<:RWMH}) = "RWMH"
transition_type(spl::Sampler{<:RWMH}) = MHTransition

function Sampler(
    alg::RWMH,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Set up VarInfo.
    vi = VarInfo(model)

    # Create a vector of prior distributions.
    space = getspace(alg)
    vns = _getvns(vi, s, Val(space))

    # Generate a sampler to retrieve the initial theta.
    spl = Sampler(alg, info, s, SamplerState(vi))

    # Initial theta.
    init_theta = vi[spl]

    # Make a sampler state, using a dummy density model.
    state = MHState(vi, AMH.RWMH(init_theta, MvNormal(zeros(length(init_theta)), 1)), AMH.DensityModel(x -> 0.0), Dict{VarName, AMH.Transition}())
    println(state.mh_sampler)

    # Generate a sampler.
    spl = Sampler(alg, info, s, state)

    # Create the actual density model.
    spl.state.density_model = AMH.DensityModel(gen_logπ(vi, spl, model))

    return spl
end

function sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Union{MH, RWMH}},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; verbose=verbose, kwargs...)

    # Convert to transformed space if we're using
    # non-Gibbs sampling.
    if !islinked(spl.state.vi, spl) && spl.selector.tag == :default
        link!(spl.state.vi, spl)
        runmodel!(model, spl.state.vi, spl)
    end
end

function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Union{MH,RWMH}},
    N::Integer;
    kwargs...
)
    mh_trans = step!(
        rng, 
        spl.state.density_model, 
        spl.state.mh_sampler, 
        N; 
        kwargs...
    )

    return MHTransition(spl, mh_trans)
end


function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    N::Integer,
    T::MHTransition;
    kwargs...
)
    # Generate a new transition.
    mh_trans = step!(
        rng, 
        spl.state.density_model, 
        spl.state.mh_sampler, 
        N, 
        T.mh_trans; 
        kwargs...
    )

    # Update the parameters in the VarInfo.
    set_vi_vals!(spl.state.vi, mh_trans.params)
    # spl.state.vi[spl] = reduce(vcat, mh_trans.params)

    return MHTransition(spl, mh_trans)
end

function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:RWMH},
    N::Integer,
    T::MHTransition;
    kwargs...
)
    # Generate a new transition.
    mh_trans = step!(
        rng, 
        spl.state.density_model, 
        spl.state.mh_sampler, 
        N, 
        T.mh_trans; 
        kwargs...
    )

    # Update the parameters in the VarInfo.
    # set_vi_vals!(spl.state.vi, mh_trans.params)
    spl.state.vi[spl] = mh_trans.params

    return MHTransition(spl, mh_trans)
end

####
#### Compiler interface, i.e. tilde operators.
####
function assume(
    spl::Sampler{<:Union{MH,RWMH}},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    Turing.DEBUG && @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "vn = $vn"
    Turing.DEBUG && @debug "r = $r" "typeof(r)=$(typeof(r))"
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function dot_assume(
    spl::Sampler{<:Union{MH,RWMH}},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi::VarInfo,
)
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function dot_assume(
    spl::Sampler{<:Union{MH,RWMH}},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi::VarInfo,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function observe(
    spl::Sampler{<:Union{MH,RWMH}},
    d::Distribution,
    value,
    vi::VarInfo,
)
    return observe(SampleFromPrior(), d, value, vi)
end

function dot_observe(
    spl::Sampler{<:Union{MH,RWMH}},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    return dot_observe(SampleFromPrior(), ds, value, vi)
end
