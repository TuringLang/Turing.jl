###
### Sampler states
###

struct MH{space,P} <: InferenceAlgorithm
    proposals::P
end

proposal(p::AdvancedMH.Proposal) = p
proposal(f::Function) = AdvancedMH.StaticProposal(f)
proposal(d::Distribution) = AdvancedMH.StaticProposal(d)
proposal(cov::AbstractMatrix) = AdvancedMH.RandomWalkProposal(MvNormal(cov))
proposal(x) = error("proposals of type ", typeof(x), " are not supported")

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

The default `MH` will draw proposal samples from the prior distribution using `AdvancedMH.StaticProposal`.

```julia
@model function gdemo(x, y)
    s² ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end

chain = sample(gdemo(1.5, 2.0), MH(), 1_000)
mean(chain)
```

Alternatively, you can specify particular parameters to sample if you want to combine sampling
from multiple samplers:

```julia
# Samples s² with MH and m with PG
chain = sample(gdemo(1.5, 2.0), Gibbs(MH(:s²), PG(10, :m)), 1_000)
mean(chain)
```

Specifying a single distribution implies the use of static MH:

```julia
# Use a static proposal for s² (which happens to be the same
# as the prior) and a static proposal for m (note that this 
# isn't a random walk proposal).
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => InverseGamma(2, 3),
        :m => Normal(0, 1)
    ),
    1_000
)
mean(chain)
```

Specifying explicit proposals using the `AdvancedMH` interface:

```julia
# Use a static proposal for s² and random walk with proposal
# standard deviation of 0.25 for m.
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => AdvancedMH.StaticProposal(InverseGamma(2,3)),
        :m => AdvancedMH.RandomWalkProposal(Normal(0, 0.25))
    ),
    1_000
)
mean(chain)
```

Using a custom function to specify a conditional distribution:

```julia
# Use a static proposal for s and and a conditional proposal for m,
# where the proposal is centered around the current sample.
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => InverseGamma(2, 3),
        :m => x -> Normal(x, 1)
    ),
    1_000
)
mean(chain)
```

Providing a covariance matrix will cause `MH` to perform random-walk
sampling in the transformed space with proposals drawn from a multivariate
normal distribution. The provided matrix must be positive semi-definite and
square:

```julia
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
            push!(props, proposal(s[2]))
        elseif length(space) == 1
            # If we hit this block, check to see if it's
            # a run-of-the-mill proposal or covariance
            # matrix.
            prop = proposal(s)

            # Return early, we got a covariance matrix.
            return MH{(),typeof(prop)}(prop)
        else
            # Try to convert it to a proposal anyways,
            # throw an error if not acceptable.
            prop = proposal(s)
            push!(props, prop)
        end
    end

    proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))
    syms = vcat(syms, prop_syms)

    return MH{tuple(syms...),typeof(proposals)}(proposals)
end

# Some of the proposals require working in unconstrained space.
transform_maybe(proposal::AMH.Proposal) = proposal
function transform_maybe(proposal::AMH.RandomWalkProposal)
    return AMH.RandomWalkProposal(Bijectors.transformed(proposal.proposal))
end

function MH(model::Model; proposal_type=AMH.StaticProposal)
    priors = DynamicPPL.extract_priors(model)
    props = Tuple([proposal_type(prop) for prop in values(priors)])
    vars = Tuple(map(Symbol, collect(keys(priors))))
    priors = map(transform_maybe, NamedTuple{vars}(props))
    return AMH.MetropolisHastings(priors)
end

#####################
# Utility functions #
#####################

"""
    set_namedtuple!(vi::VarInfo, nt::NamedTuple)

Places the values of a `NamedTuple` into the relevant places of a `VarInfo`.
"""
function set_namedtuple!(vi::DynamicPPL.VarInfoOrThreadSafeVarInfo, nt::NamedTuple)
    # TODO: Replace this with something like
    # for vn in keys(vi)
    #     vi = DynamicPPL.setindex!!(vi, get(nt, vn))
    # end
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
const MHLogDensityFunction{M<:Model,S<:Sampler{<:MH},V<:AbstractVarInfo} = Turing.LogDensityFunction{
    V,M,<:DynamicPPL.SamplingContext{<:S}
}

function LogDensityProblems.logdensity(f::MHLogDensityFunction, x::NamedTuple)
    # TODO: Make this work with immutable `f.varinfo` too.
    sampler = DynamicPPL.getsampler(f)
    vi = f.varinfo

    x_old, lj_old = vi[sampler], getlogp(vi)
    set_namedtuple!(vi, x)
    vi_new = last(DynamicPPL.evaluate!!(f.model, vi, DynamicPPL.getcontext(f)))
    lj = getlogp(vi_new)

    # Reset old `vi`.
    setindex!!(vi, x_old, sampler)
    setlogp!!(vi, lj_old)

    return lj
end

# unpack a vector if possible
unvectorize(dists::AbstractVector) = length(dists) == 1 ? first(dists) : dists

# possibly unpack and reshape samples according to the prior distribution
reconstruct(dist::Distribution, val::AbstractVector) = DynamicPPL.reconstruct(dist, val)
function reconstruct(dist::AbstractVector{<:UnivariateDistribution}, val::AbstractVector)
    return val
end
function reconstruct(dist::AbstractVector{<:MultivariateDistribution}, val::AbstractVector)
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
    dist_val_tuple(spl::Sampler{<:MH}, vi::VarInfo)

Return two `NamedTuples`.

The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::Sampler{<:MH}, vi::DynamicPPL.VarInfoOrThreadSafeVarInfo)
    vns = _getvns(vi, spl)
    dt = _dist_tuple(spl.alg.proposals, vi, vns)
    vt = _val_tuple(vi, vns)
    return dt, vt
end

@generated function _val_tuple(vi::VarInfo, vns::NamedTuple{names}) where {names}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :(
            $name = reconstruct(
                unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name)),
                DynamicPPL.getval(vi, vns.$name),
            )
        ) for name in names
    ]
    return expr
end

@generated function _dist_tuple(
    props::NamedTuple{propnames}, vi::VarInfo, vns::NamedTuple{names}
) where {names,propnames}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        if name in propnames
            # We've been given a custom proposal, use that instead.
            :($name = props.$name)
        else
            # Otherwise, use the default proposal.
            :(
                $name = AMH.StaticProposal(
                    unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name))
                )
            )
        end for name in names
    ]
    return expr
end

# Utility functions to link
should_link(varinfo, sampler, proposal) = false
function should_link(varinfo, sampler, proposal::NamedTuple{(),Tuple{}})
    # If it's an empty `NamedTuple`, we're using the priors as proposals
    # in which case we shouldn't link.
    return false
end
function should_link(varinfo, sampler, proposal::AdvancedMH.RandomWalkProposal)
    return true
end
# FIXME: This won't be hit unless `vals` are all the exactly same concrete type of `AdvancedMH.RandomWalkProposal`!
function should_link(
    varinfo, sampler, proposal::NamedTuple{names,vals}
) where {names,vals<:NTuple{<:Any,<:AdvancedMH.RandomWalkProposal}}
    return true
end

function maybe_link!!(varinfo, sampler, proposal, model)
    return if should_link(varinfo, sampler, proposal)
        link!!(varinfo, sampler, model)
    else
        varinfo
    end
end

# Make a proposal if we don't have a covariance proposal matrix (the default).
function propose!!(
    rng::AbstractRNG, vi::AbstractVarInfo, model::Model, spl::Sampler{<:MH}, proposal
)
    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl, vi)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, getlogp(vi), false)

    # Make a new transition.
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            Turing.LogDensityFunction(
                vi,
                model,
                DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
            ),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)

    # TODO: Make this compatible with immutable `VarInfo`.
    # Update the values in the VarInfo.
    set_namedtuple!(vi, trans.params)
    return setlogp!!(vi, trans.lp)
end

# Make a proposal if we DO have a covariance proposal matrix.
function propose!!(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::Model,
    spl::Sampler{<:MH},
    proposal::AdvancedMH.RandomWalkProposal,
)
    # If this is the case, we can just draw directly from the proposal
    # matrix.
    vals = vi[spl]

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(spl.alg.proposals)
    prev_trans = AMH.Transition(vals, getlogp(vi), false)

    # Make a new transition.
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            Turing.LogDensityFunction(
                vi,
                model,
                DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
            ),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)

    return setlogp!!(DynamicPPL.unflatten(vi, spl, trans.params), trans.lp)
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:MH},
    vi::AbstractVarInfo;
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vi = maybe_link!!(vi, spl, spl.alg.proposals, model)

    return Transition(model, vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, spl::Sampler{<:MH}, vi::AbstractVarInfo; kwargs...
)
    # Cases:
    # 1. A covariance proposal matrix
    # 2. A bunch of NamedTuples that specify the proposal space
    vi = propose!!(rng, vi, model, spl, spl.alg.proposals)

    return Transition(model, vi), vi
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(rng, spl::Sampler{<:MH}, dist::Distribution, vn::VarName, vi)
    DynamicPPL.updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn)), vi
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
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1]))), vi
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:MH},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1]))), vi
end

function DynamicPPL.observe(spl::Sampler{<:MH}, d::Distribution, value, vi)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:MH},
    ds::Union{Distribution,AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end
