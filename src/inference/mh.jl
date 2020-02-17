###
### Sampler states
###

struct MH{space, P} <: InferenceAlgorithm 
    proposals::P
end

function MH(space...)
    syms = Symbol[]

    prop_syms = Symbol[]
    props = AMH.Proposal[]

    check_support(dist) = insupport(dist, z)

    for s in space
        if s isa Symbol
            push!(syms, s)
        elseif s isa Pair || s isa Tuple
            push!(prop_syms, s[1])

            if s[2] isa AMH.Proposal
                push!(props, s[2])
            elseif s[2] isa Distribution
                push!(props, AMH.Proposal(AMH.Static(), s[2]))
            elseif s[2] isa Function
                push!(props, AMH.Proposal(AMH.Static(), s[2]))
            end
        end
    end

    proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))
    syms = vcat(syms, prop_syms)
    return MH{tuple(syms...), typeof(proposals)}(proposals)
end

alg_str(::Sampler{<:MH}) = "MH"

#################
# MH Transition #
#################

struct MHTransition{T, F<:AbstractFloat, M<:AMH.Transition}
    θ    :: T
    lp   :: F
    mh_trans :: M
end

function MHTransition(spl::Sampler{<:MH}, mh_trans::AMH.Transition)
    theta = tonamedtuple(spl.state.vi)
    return MHTransition(theta, mh_trans.lp, mh_trans)
end

#####################
# Utility functions #
#####################

"""
    set_namedtuple!(vi::VarInfo, nt::NamedTuple)

Places the values of a `NamedTuple` into the relevant places of a `VarInfo`.
"""
function set_namedtuple!(vi::VarInfo, nt::NamedTuple)
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns

        n_vns = length(vns)
        n_vals = length(vals)
        v_isarr = vals isa AbstractArray

        if v_isarr && n_vals == 1 && n_vns > 1
            for (vn, val) in zip(vns, vals[1])
                vi[vn] = val isa AbstractArray ? val : [val]
            end
        elseif v_isarr && n_vals > 1 && n_vns == 1
            vi[vns[1]] = vals
        elseif v_isarr && n_vals == 1 && n_vns == 1
            if vals[1] isa AbstractArray
                vi[vns[1]] = vals[1]
            else
                vi[vns[1]] = [vals[1]]
            end
        elseif !(v_isarr)
            vi[vns[1]] = [vals]
        else
            error("Cannot assign `NamedTuple` to `VarInfo`")
        end
    end
end

"""
    gen_logπ_mh(vi::VarInfo, spl::Sampler, model)   

Generate a log density function -- this variant uses the 
`set_namedtuple!` function to update the `VarInfo`.
"""
function gen_logπ_mh(spl::Sampler, model)
    function logπ(x)::Float64
        vi = spl.state.vi
        x_old, lj_old = vi[spl], getlogp(vi)
        # vi[spl] = x
        set_namedtuple!(vi, x)
        runmodel!(model, vi)
        lj = getlogp(vi)
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

"""
    dist_val_tuple(spl::Sampler{<:MH})

Returns two `NamedTuples`. The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::Sampler{<:MH})
    vns = _getvns(spl.state.vi, spl)
    dt = _dist_tuple(spl.state.vi.metadata, spl.alg.proposals, spl.state.vi, vns)
    vt = _val_tuple(spl.state.vi.metadata, spl.state.vi, vns)
    return dt, vt
end

@generated function _val_tuple(metadata::NamedTuple, vi::VarInfo, vns::NamedTuple{names}) where {names}
    length(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(
            length(metadata.$f.vns) == 1 ? getindex(vi, metadata.$f.vns)[1] : getindex.(Ref(vi), metadata.$f.vns)
        )))
    end
    return expr
end

@generated function _dist_tuple(
    metadata::NamedTuple, 
    props::NamedTuple{propnames}, 
    vi::VarInfo, 
    vns::NamedTuple{names}
) where {names, propnames}
    length(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    map(names) do f
        if f in propnames
            # We've been given a custom proposal, use that instead.
            push!(expr.args, Expr(:(=), f, :(props.$f)))
        else
            # Otherwise, use the default proposal.
            push!(expr.args, Expr(:(=), f, :(AMH.Proposal(AMH.Static(), metadata.$f.dists[1]))))
        end
    end
    return expr
end

#################
# Sampler state #
#################

mutable struct MHState{V<:VarInfo} <: AbstractSamplerState
    vi :: V
    density_model :: AMH.DensityModel
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

    # Make a varinfo.
    vi = VarInfo(model)

    # Make a density model.
    dm = AMH.DensityModel(x -> 0.0)

    # Set up state struct.
    state = MHState(vi, dm)

    # Generate a sampler.
    spl = Sampler(alg, info, s, state)

    # Update the density model.
    spl.state.density_model = AMH.DensityModel(gen_logπ_mh(spl, model))

    return spl
end

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; verbose=verbose, kwargs...)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    N::Integer,
    transition;
    kwargs...
)
    if spl.selector.rerun # Recompute joint in logp
        runmodel!(model, spl.state.vi)
    end

    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, getlogp(spl.state.vi))

    # Make a new transition.
    trans = AbstractMCMC.step!(rng, spl.state.density_model, mh_sampler, 1, prev_trans)

    # Update the values in the VarInfo.
    set_namedtuple!(spl.state.vi, trans.params)
    setlogp!(spl.state.vi, trans.lp)

    return Transition(spl)
end

####
#### Compiler interface, i.e. tilde operators.
####
function assume(
    spl::Sampler{<:MH},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function dot_assume(
    spl::Sampler{<:MH},
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
    spl::Sampler{<:MH},
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
    spl::Sampler{<:MH},
    d::Distribution,
    value,
    vi::VarInfo,
)
    return observe(SampleFromPrior(), d, value, vi)
end

function dot_observe(
    spl::Sampler{<:MH},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    return dot_observe(SampleFromPrior(), ds, value, vi)
end
