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

drop_space(alg::MH{space,P}) where {space,P} = MH{(),P}(alg.proposals)

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
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns
        if vals isa AbstractVector
            vals = unvectorize(vals)
        end
        if length(vns) == 1
            # Only one variable, assign the values to it
            DynamicPPL.setindex!(vi, vals, vns[1])
        else
            # Spread the values across the variables
            length(vns) == length(vals) || error("Unequal number of variables and values")
            for (vn, val) in zip(vns, vals)
                DynamicPPL.setindex!(vi, val, vn)
            end
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
function reconstruct(dist::Distribution, val::AbstractVector)
    return DynamicPPL.from_vec_transform(dist)(val)
end
reconstruct(dist::AbstractVector{<:UnivariateDistribution}, val::AbstractVector) = val
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
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :(
            $name = reconstruct(
                unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name)),
                DynamicPPL.getindex_internal(vi, vns.$name),
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
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Sampler{<:MH}, dist::Distribution, vn::VarName, vi
)
    # Just defer to `SampleFromPrior`.
    retval = DynamicPPL.assume(rng, SampleFromPrior(), dist, vn, vi)
    # Update the Gibbs IDs because they might have been assigned in the `SampleFromPrior` call.
    DynamicPPL.updategid!(vi, vn, spl)
    # Return.
    return retval
end

function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:MH},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::AbstractVarInfo,
)
    # Just defer to `SampleFromPrior`.
    retval = DynamicPPL.dot_assume(rng, SampleFromPrior(), dist, vns[1], var, vi)
    # Update the Gibbs IDs because they might have been assigned in the `SampleFromPrior` call.
    DynamicPPL.updategid!.((vi,), vns, (spl,))
    # Return.
    return retval
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:MH},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::AbstractVarInfo,
)
    # Just defer to `SampleFromPrior`.
    retval = DynamicPPL.dot_assume(rng, SampleFromPrior(), dists, vns, var, vi)
    # Update the Gibbs IDs because they might have been assigned in the `SampleFromPrior` call.
    DynamicPPL.updategid!.((vi,), vns, (spl,))
    return retval
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
