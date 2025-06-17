###
### Sampler states
###

proposal(p::AdvancedMH.Proposal) = p
proposal(f::Function) = AdvancedMH.StaticProposal(f)
proposal(d::Distribution) = AdvancedMH.StaticProposal(d)
proposal(cov::AbstractMatrix) = AdvancedMH.RandomWalkProposal(MvNormal(cov))
proposal(x) = error("proposals of type ", typeof(x), " are not supported")

"""
    MH(proposals...)

Construct a Metropolis-Hastings algorithm.

The arguments `proposals` can be

- Blank (i.e. `MH()`), in which case `MH` defaults to using the prior for each parameter as the proposal distribution.
- An iterable of pairs or tuples mapping a `Symbol` to a `AdvancedMH.Proposal`, `Distribution`, or `Function`
  that returns a conditional proposal distribution.
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
struct MH{P} <: AbstractSampler
    proposals::P

    function MH(proposals...)
        prop_syms = Symbol[]
        props = AMH.Proposal[]

        for s in proposals
            if s isa Pair || s isa Tuple
                # Check to see whether it's a pair that specifies a kernel
                # or a specific proposal distribution.
                push!(prop_syms, s[1])
                push!(props, proposal(s[2]))
            elseif length(proposals) == 1
                # If we hit this block, check to see if it's
                # a run-of-the-mill proposal or covariance
                # matrix.
                prop = proposal(s)

                # Return early, we got a covariance matrix.
                return new{typeof(prop)}(prop)
            else
                # Try to convert it to a proposal anyways,
                # throw an error if not acceptable.
                prop = proposal(s)
                push!(props, prop)
            end
        end

        proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))

        return new{typeof(proposals)}(proposals)
    end
end

# Turing sampler interface
DynamicPPL.initialsampler(::MH) = DynamicPPL.SampleFromPrior()
get_adtype(::MH) = nothing
update_sample_kwargs(::MH, ::Integer, kwargs) = kwargs
requires_unconstrained_space(::MH) = false
requires_unconstrained_space(::MH{<:AdvancedMH.RandomWalkProposal}) = true
# `NamedTuple` of proposals. TODO: It seems, at some point, that there
# was an intent to extract the parameters from the NamedTuple and to only
# link those parameters that corresponded to RandomWalkProposals. See
# https://github.com/TuringLang/Turing.jl/issues/1583.
requires_unconstrained_space(::MH{NamedTuple{(),Tuple{}}}) = false
@generated function requires_unconstrained_space(
    ::MH{<:NamedTuple{names,props}}
) where {names,props}
    # If we have a `NamedTuple` with proposals, we check if all of them are
    # `AdvancedMH.RandomWalkProposal`. If so, we need to link.
    return all(prop -> prop <: AdvancedMH.RandomWalkProposal, props.parameters)
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

This variant uses the `set_namedtuple!` function to update the `VarInfo`.
"""
const MHLogDensityFunction{M<:Model,S<:MH,V<:AbstractVarInfo} =
    DynamicPPL.LogDensityFunction{M,V,<:DynamicPPL.SamplingContext{<:S},AD} where {AD}

function LogDensityProblems.logdensity(f::MHLogDensityFunction, x::NamedTuple)
    vi = deepcopy(f.varinfo)
    set_namedtuple!(vi, x)
    vi_new = last(DynamicPPL.evaluate!!(f.model, vi, f.context))
    lj = getlogp(vi_new)
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
    dist_val_tuple(spl::MH, vi::VarInfo)

Return two `NamedTuples`.

The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::MH, vi::DynamicPPL.VarInfoOrThreadSafeVarInfo)
    vns = all_varnames_grouped_by_symbol(vi)
    dt = _dist_tuple(spl.proposals, vi, vns)
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
_val_tuple(::VarInfo, ::Tuple{}) = ()

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
_dist_tuple(::@NamedTuple{}, ::VarInfo, ::Tuple{}) = ()

# Make a proposal if we don't have a covariance proposal matrix (the default).
function propose!!(
    rng::AbstractRNG, vi::AbstractVarInfo, ldf::LogDensityFunction, spl::MH, proposal
)
    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl, vi)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, getlogp(vi), false)

    # Make a new transition.
    densitymodel = AMH.DensityModel(Base.Fix1(LogDensityProblems.logdensity, ldf))
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
    ldf::LogDensityFunction,
    spl::MH,
    proposal::AdvancedMH.RandomWalkProposal,
)
    # If this is the case, we can just draw directly from the proposal
    # matrix.
    vals = vi[:]

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(spl.proposals)
    prev_trans = AMH.Transition(vals, getlogp(vi), false)

    # Make a new transition.
    densitymodel = AMH.DensityModel(Base.Fix1(LogDensityProblems.logdensity, ldf))
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)

    return setlogp!!(DynamicPPL.unflatten(vi, trans.params), trans.lp)
end

function AbstractMCMC.step(rng::AbstractRNG, ldf::LogDensityFunction, spl::MH; kwargs...)
    vi = ldf.varinfo
    return Transition(ldf.model, vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, ldf::LogDensityFunction, spl::MH, vi::AbstractVarInfo; kwargs...
)
    vi = propose!!(rng, vi, ldf, spl, spl.proposals)
    return Transition(ldf.model, vi), vi
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    rng::Random.AbstractRNG, ::MH, dist::Distribution, vn::VarName, vi
)
    # Just defer to `SampleFromPrior`.
    return DynamicPPL.assume(rng, SampleFromPrior(), dist, vn, vi)
end

function DynamicPPL.observe(::MH, d::Distribution, value, vi)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end
