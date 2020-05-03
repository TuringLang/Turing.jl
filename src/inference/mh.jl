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

    for s in space
        if s isa Symbol
            push!(syms, s)
        elseif s isa Pair || s isa Tuple
            push!(prop_syms, s[1])

            if s[2] isa AMH.Proposal
                push!(props, s[2])
            elseif s[2] isa Distribution
                push!(props, AMH.StaticProposal(s[2]))
            elseif s[2] isa Function
                push!(props, AMH.StaticProposal(s[2]))
            end
        end
    end

    proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))
    syms = vcat(syms, prop_syms)
    return MH{tuple(syms...), typeof(proposals)}(proposals)
end

function Sampler(
    alg::MH,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Set up state struct.
    state = SamplerState(VarInfo(model))

    # Generate a sampler.
    return Sampler(alg, info, s, state)
end

alg_str(::Sampler{<:MH}) = "MH"

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
        elseif v_isarr && n_vals == n_vns > 1
            for (vn, val) in zip(vns, vals)
                vi[vn] = [val]
            end
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
    MHLogDensityFunction

A log density function for the MH sampler.

This variant uses the  `set_namedtuple!` function to update the `VarInfo`.
"""
struct MHLogDensityFunction{M<:Model,S<:Sampler{<:MH}} <: Function # Relax AMH.DensityModel?
    model::M
    sampler::S
end

function (f::MHLogDensityFunction)(x)::Float64
    sampler = f.sampler
    vi = sampler.state.vi
    x_old, lj_old = vi[sampler], getlogp(vi)
    # vi[sampler] = x
    set_namedtuple!(vi, x)
    f.model(vi)
    lj = getlogp(vi)
    vi[sampler] = x_old
    setlogp!(vi, lj_old)
    return lj
end

# unpack a vector if possible
unvectorize(dists::AbstractVector) = length(dists) == 1 ? first(dists) : dists

# possibly unpack and reshape samples according to the prior distribution 
reconstruct(dist::Distribution, val::AbstractVector) = DynamicPPL.reconstruct(dist, val)
function reconstruct(
    dist::AbstractVector{<:UnivariateDistribution},
    val::AbstractVector
)
    return val
end

"""
    dist_val_tuple(spl::Sampler{<:MH})

Returns two `NamedTuples`. The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::Sampler{<:MH})
    vi = spl.state.vi
    vns = _getvns(vi, spl)
    dt = _dist_tuple(spl.alg.proposals, vi, vns)
    vt = _val_tuple(vi, vns)
    return dt, vt
end

@generated function _val_tuple(
    vi::VarInfo,
    vns::NamedTuple{names}
) where {names}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :($name = reconstruct(unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name)),
                              DynamicPPL.getval(vi, vns.$name)))
        for name in names]
    return expr
end

@generated function _dist_tuple(
    props::NamedTuple{propnames}, 
    vi::VarInfo,
    vns::NamedTuple{names}
) where {names,propnames}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        if name in propnames
            # We've been given a custom proposal, use that instead.
            :($name = props.$name)
        else
            # Otherwise, use the default proposal.
            :($name = AMH.StaticProposal(unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name))))
        end for name in names]
    return expr
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
        model(spl.state.vi)
    end

    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, getlogp(spl.state.vi))

    # Make a new transition.
    densitymodel = AMH.DensityModel(MHLogDensityFunction(model, spl))
    trans = AbstractMCMC.step!(rng, densitymodel, mh_sampler, 1, prev_trans)

    # Update the values in the VarInfo.
    set_namedtuple!(spl.state.vi, trans.params)
    setlogp!(spl.state.vi, trans.lp)

    return Transition(spl)
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    spl::Sampler{<:MH},
    dist::Distribution,
    vn::VarName,
    vi,
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function DynamicPPL.dot_assume(
    spl::Sampler{<:MH},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi,
)
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function DynamicPPL.dot_assume(
    spl::Sampler{<:MH},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function DynamicPPL.observe(
    spl::Sampler{<:MH},
    d::Distribution,
    value,
    vi,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:MH},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end
