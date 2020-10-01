###
### Sampler states
###

struct MH{space, P} <: InferenceAlgorithm 
    proposals::P
end

proposal(p::AdvancedMH.Proposal) = p
proposal(cov::AbstractMatrix) = AdvancedMH.RandomWalkProposal(MvNormal(cov))

"""
    MH(space...)

Construct a Metropolis-Hastings algorithm. 

The arguments `space` can be 

- Blank (i.e. `MH()`), in which case `MH` defaults to using the prior for each parameter as the proposal distribution.
- A set of one or more symbols to sample with `MH` in conjunction with `Gibbs`, i.e. `Gibbs(MH(:m), PG(10, :s))`
- An iterable of pairs or tuples mapping a `Symbol` to a `AdvancedMH.Proposal`, `Distribution`, or `Function` 
  that generates returns a conditional proposal distribution.
- A covariance matrix to use as for mean-zero multivariate normal proposals.

# Examples

The default `MH` will use propose samples from the prior distribution using `AdvancedMH.StaticProposal`.

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

chain = sample(gdemo(1.5, 2.0), MH(), 1_000)
mean(chain)
```

Alternatively, you can specify particular parameters to sample if you want to combine sampling
from multiple samplers:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Samples s with MH and m with PG
chain = sample(gdemo(1.5, 2.0), Gibbs(MH(:s), PG(10, :m)), 1_000)
mean(chain)
```

Using custom distributions defaults to using static MH:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Use a static proposal for s and random walk with proposal 
# standard deviation of 0.25 for m.
chain = sample(
    gdemo(1.5, 2.0), 
    MH(
        :s => InverseGamma(2, 3),
        :m => Normal(0, 1)
    ), 
    1_000
)
mean(chain)
```

Specifying explicit proposals using the `AdvancedMH` interface:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Use a static proposal for s and random walk with proposal 
# standard deviation of 0.25 for m.
chain = sample(
    gdemo(1.5, 2.0), 
    MH(
        :s => AdvancedMH.StaticProposal(InverseGamma(2,3)),
        :m => AdvancedMH.RandomWalkProposal(Normal(0, 0.25))
    ), 
    1_000
)
mean(chain)
```

Using a custom function to specify a conditional distribution:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Use a static proposal for s and and a conditional proposal for m,
# where the proposal is centered around the current sample.
chain = sample(
    gdemo(1.5, 2.0), 
    MH(
        :s => InverseGamma(2, 3),
        :m => x -> Normal(x, 1)
    ), 
    1_000
)
mean(chain)
```

Providing a covariance matrix will cause `MH` to perform random-walk
sampling in the transformed space with proposals drawn from a multivariate
normal distribution. The provided matrix must be positive semi-definite and square. Usage:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Providing a custom variance-covariance matrix
chain = sample(
    gdemo(1.5, 2.0), 
    MH(
        [0.25 0.05; 
         0.05 0.50]
    ), 
    1_000
)
mean(chain)
```

"""
function MH(space...)
    syms = Symbol[]

    prop_syms = Symbol[]
    props = AMH.Proposal[]

    for s in space
        if s isa Symbol
            # If it's just a symbol, proceed as normal.
            push!(syms, s)
        elseif s isa Pair || s isa Tuple
            # Check to see whether it's a pair that specifies a kernel
            # or a specific proposal distribution.
            push!(prop_syms, s[1])

            if s[2] isa AMH.Proposal
                push!(props, s[2])
            elseif s[2] isa Distribution
                push!(props, AMH.StaticProposal(s[2]))
            elseif s[2] isa Function
                push!(props, AMH.StaticProposal(s[2]))
            end
        elseif length(space) == 1
            # If we hit this block, check to see if it's 
            # a run-of-the-mill proposal or covariance
            # matrix.
            prop = proposal(s)

            # Return early, we got a covariance matrix. 
            return MH{(), typeof(prop)}(prop)
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

isgibbscomponent(::MH) = true

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
        nvns = length(vns)

        # if there is a single variable only
        if nvns == 1
            # assign the unpacked values
            if length(vals) == 1
                vi[vns[1]] = [vals[1];]
            # otherwise just assign the values
            else
                vi[vns[1]] = [vals;]
            end
        # if there are multiple variables
        elseif vals isa AbstractArray
            nvals = length(vals)
            # if values are provided as an array with a single element
            if nvals == 1
                # iterate over variables and unpacked values
                for (vn, val) in zip(vns, vals[1])
                    vi[vn] = [val;]
                end
            # otherwise number of variables and number of values have to be equal
            elseif nvals == nvns
                # iterate over variables and values
                for (vn, val) in zip(vns, vals)
                    vi[vn] = [val;]
                end
            else
                error("Cannot assign `NamedTuple` to `VarInfo`")
            end
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
function reconstruct(
    dist::AbstractVector{<:MultivariateDistribution},
    val::AbstractVector
)
    offset = 0
    return map(dist) do d
        n = length(d)
        newoffset = offset + n
        v = val[(offset + 1):newoffset]
        offset = newoffset
        return v
    end
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

# Utility functions to link or 
maybe_link!(varinfo, sampler, proposal) = nothing
function maybe_link!(varinfo, sampler, proposal::AdvancedMH.RandomWalkProposal)
    link!(varinfo, sampler)
end

maybe_invlink!(varinfo, sampler, proposal) = nothing
function maybe_invlink!(varinfo, sampler, proposal::AdvancedMH.RandomWalkProposal)
    invlink!(varinfo, sampler)
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

    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    maybe_link!(spl.state.vi, spl, spl.alg.proposals)
end

function AbstractMCMC.sample_end!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    N::Integer,
    transitions;
    kwargs...
)
    # We are doing a random walk, so we unlink everything when we're done.
    maybe_invlink!(spl.state.vi, spl, spl.alg.proposals)

end

# Make a proposal if we don't have a covariance proposal matrix (the default).
function propose!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    proposal
)
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
end

# Make a proposal if we DO have a covariance proposal matrix.
function propose!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    proposal::AdvancedMH.RandomWalkProposal{<:MvNormal}
)
    # If this is the case, we can just draw directly from the proposal
    # matrix.
    vals = spl.state.vi[spl]

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(spl.alg.proposals)
    prev_trans = AMH.Transition(vals, getlogp(spl.state.vi))

    # Make a new transition.
    densitymodel = AMH.DensityModel(gen_logπ(spl.state.vi, spl, model))
    trans = AbstractMCMC.step!(rng, densitymodel, mh_sampler, 1, prev_trans)

    # Update the values in the VarInfo.
    spl.state.vi[spl] = trans.params
    setlogp!(spl.state.vi, trans.lp)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{MH{space, P}},
    N::Integer,
    transition;
    kwargs...
) where {space, P}
    if spl.selector.rerun # Recompute joint in logp
        model(spl.state.vi)
    end

    # Cases:
    # 1. A covariance proposal matrix
    # 2. A bunch of NamedTuples that specify the proposal space
    propose!(rng, model, spl, spl.alg.proposals)

    return Transition(spl)
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    rng,
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
    rng,
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
    rng,
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
