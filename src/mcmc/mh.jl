using AdvancedMH: AdvancedMH
using BangBang: BangBang

"""
    MH([proposal])

Construct a Metropolis-Hastings algorithm.

The argument `proposal` can be

- Blank (i.e. `MH()`), in which case `MH` defaults to using the prior for each parameter as
  the proposal distribution.
- A mapping of `VarName`s to a `Distribution`, or generic callable that returns a
  conditional proposal distribution.
- A covariance matrix to use as for mean-zero multivariate normal proposals. Note that if a
  covariance matrix is passed, sampling occurs in linked space, so the size of an MH step
  may differ from the size of a step in parameter space.

# Examples

Consider the model below:

```julia
@model function gdemo()
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
end
```

The default constructor, `MH()`, uses the prior distributions as proposals. So, new
proposals are obtained by sampling `s` from `InverseGamma(2,3)` and `m` from `Normal(0,
sqrt(s))`.

```julia
spl = MH()
```

Alternatively, a mapping of variable names to proposal distributions can be provided.
This implies the use of static proposals for each variable. If a variable is not specified,
its prior distribution is used as the proposal.

```julia
# Use a static proposal for s² (which happens to be the same as the prior) and a static
# proposal for m (note that this isn't a random walk proposal).
spl = MH(
    # This happens to be the same as the prior
    @varname(s) => InverseGamma(2, 3),
    # This is different from the prior
    @varname(m) => Normal(0, 1),
)
```

If the `VarName` of interest is a single symbol, you can also use a `Symbol` instead.

```julia
spl = MH(
    :s => InverseGamma(2, 3),
    :m => Normal(0, 1),
)
```

You can also use a callable to define a proposal that is conditional on the current values.
The callable must accept a single argument, which is a `DynamicPPL.VarNamedTuple` that holds
all the values of the parameters from the previous step. You can obtain the value of a
specific parameter by indexing into this `VarNamedTuple` using a `VarName` (note that symbol
indexing is not supported). The callable must then return a `Distribution` from which to
draw the proposal.

!!! note
    In general, there is no way for Turing to reliably detect whether a proposal is meant to
    be a callable or not, since callable structs may have any type. Hence, any proposal that
    is *not* a distribution is assumed to be a callable.

```julia
spl = MH(
    # This is a static proposal (same as above).
    @varname(s) => InverseGamma(2, 3),
    # This is a conditional proposal, which proposes m from a normal
    # distribution centred at the current value of m, with a standard
    # deviation of 0.5.
    @varname(m) => (vnt -> Normal(vnt[@varname(m)], 0.5)),
)
```

**Note that when using conditional proposals, the values obtained by indexing into the
`VarNamedTuple` are always in unlinked space.** Sometimes, you may want to define a random-walk
proposal in linked space. For this, you can use `LinkedRW` as a proposal, which takes a
covariance matrix as an argument:

```julia
spl = MH(
    @varname(s) => InverseGamma(2, 3),
    @varname(m) => LinkedRW(Diagonal([0.25]))
)
```

In the above example, `LinkedRW(Diagonal([0.25]))` defines a random-walk proposal for `m` in
linked space. This is in fact the same as the conditional proposal above, because `m` is
already unconstrained, and so linked space and unlinked space are the same for this
variable. However, `s` is constrained to be positive, and so using a `LinkedRW` proposal for
`s` would be different from using a normal proposal in unlinked space (`LinkedRW` will
ensure that the proposals for `s` always remain positive in unlinked space).

```julia
spl = MH(
    @varname(s) => LinkedRW(Diagonal([0.5])),
    @varname(m) => LinkedRW(Diagonal([0.25])),
)
```

Finally, providing just a single covariance matrix will cause `MH` to perform random-walk
sampling in linked space with proposals drawn from a multivariate normal distribution. All
variables are linked in this case. The provided matrix must be positive semi-definite and
square. This example is therefore equivalent to the previous one:

```julia
# Providing a custom variance-covariance matrix
spl = MH(
    [0.50 0;
     0 0.25]
)
```
"""
struct MH{I} <: AbstractSampler
    "A function which takes two arguments: (1) the VarNamedTuple of raw values at the
    previous step, and (2) a VarNamedTuple of linked values for any variables that have
    `LinkedRW` proposals; and returns an AbstractInitStrategy. We don't have access to the
    VNTs until the actual sampling, so we have to use a function here."
    init_strategy_constructor::I
    "Linked variables, i.e., variables which have a `LinkedRW` proposal."
    linkedrw_vns::Set{VarName}
end
MH() = MH(vnt -> DynamicPPL.InitFromPrior(), Set{VarName}())

"""
    LinkedRW(cov_matrix)

Define a random-walk proposal in linked space with the given covariance matrix. Note that
the covariance matrix must correspond exactly to the size of the variable in linked space.
"""
struct LinkedRW
    # TODO(penelopeysm): Use PDMats to check?
    "The covariance matrix to use for the random-walk proposal in linked space."
    cov_matrix::AbstractMatrix
end

struct InitFromProposals{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    "A mapping of VarNames to Tuple{Bool,Distribution}s that they should be sampled from. If
    the VarName is not in this VarNamedTuple, then it will be sampled from the prior. The
    Bool indicates whether the proposal is in linked space (true, i.e., the strategy should
    return a `LinkedVectorValue`); or in untransformed space (false, i.e., the strategy
    should return an `UntransformedValue`)."
    proposals::V
    "A cache of the prior distributions for any variables that were not given an explicit
    proposal. This is needed to compute the proposal density during MH steps."
    priors::Dict{VarName,Distribution}
end
function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, prior::Distribution, strategy::InitFromProposals
)
    if haskey(strategy.proposals, vn)
        # this is the proposal that the user wanted
        is_linkedrw, dist = strategy.proposals[vn]
        if is_linkedrw
            # LinkedRW proposals end up here.
            transform = DynamicPPL.from_linked_vec_transform(prior)
            linked_vec = rand(rng, dist)
            return DynamicPPL.UntransformedValue(transform(linked_vec))
        else
            # Static or conditional proposal in untransformed space.
            return DynamicPPL.UntransformedValue(rand(rng, dist))
        end
    else
        # No proposal was specified for this variable, so we sample from the prior. We
        # also need to cache the prior for later use in log-proposal density calculations.
        strategy.priors[vn] = prior
        return DynamicPPL.UntransformedValue(rand(rng, prior))
    end
end

const SymOrVNPair = Pair{<:Union{Symbol,VarName},<:Any}

function MH(pair1::SymOrVNPair, pairs::Vararg{SymOrVNPair})
    vn_proposal_pairs = (pair1, pairs...)
    # It is assumed that `raw_vals` is a VarNamedTuple that has all the variables' values
    # already set. Furthermore, `linked_vals` is a VarNamedTuple that has the linked values
    # for any variables that have `LinkedRW` proposals.
    function init_strategy_constructor(raw_vals, linked_vals)
        proposals = DynamicPPL.VarNamedTuple()
        for pair in vn_proposal_pairs
            vn, proposal = pair
            # Convert all keys to VarNames.
            if vn isa Symbol
                vn = DynamicPPL.VarName{vn}()
            elseif !(vn isa DynamicPPL.VarName)
                throw(
                    ArgumentError("first element of each pair must be a Symbol or VarName")
                )
            end
            # Check whether the proposal is a Distribution.
            proposal_dist = if proposal isa Distribution
                (false, proposal)
            elseif proposal isa LinkedRW
                # The distribution we draw from is an MvNormal, centred at the current
                # linked value, and with the given covariance matrix. We also need to add a
                # flag to signal that this is being sampled in linked space.
                # `linked_vals[vn]` is a MHLinkedVal struct (defined below)
                (true, MvNormal(linked_vals[vn].val, proposal.cov_matrix))
            else
                # It's a callable that takes `vnt` and returns a distribution.
                (false, proposal(raw_vals))
            end
            proposals = DynamicPPL.templated_setindex!!(
                proposals, proposal_dist, vn, raw_vals.data[AbstractPPL.getsym(vn)]
            )
        end
        return InitFromProposals(proposals, Dict{VarName,Distribution}())
    end
    return MH(
        init_strategy_constructor,
        Set{VarName}(vn for (vn, proposal) in vn_proposal_pairs if proposal isa LinkedRW),
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH;
    initial_params::DynamicPPL.AbstractInitStrategy,
    discard_sample=false,
    kwargs...,
)
    # Generate and return initial parameters. We need to use VAIMAcc because that will
    # generate the VNT for us that provides the values (as opposed to `vi.values` which
    # stores `AbstractTransformedValues`).
    #
    # TODO(penelopeysm): This in fact could very well be OnlyAccsVarInfo. Indeed, if you
    # only run MH, OnlyAccsVarInfo already works right now. The problem is that using MH
    # inside Gibbs needs a full VarInfo.
    vi = DynamicPPL.VarInfo()
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    vi = DynamicPPL.setacc!!(vi, MHLinkedValuesAccumulator(spl.linkedrw_vns))
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params)
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=true))
    return transition, vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH,
    old_vi::DynamicPPL.AbstractVarInfo;
    discard_sample=false,
    kwargs...,
)
    old_lp = DynamicPPL.getlogjoint(old_vi)
    # Get the init strategy for this step from the VAIMAcc.
    old_raw_values = DynamicPPL.getacc(old_vi, Val(:ValuesAsInModel)).values
    old_linked_values = DynamicPPL.getacc(old_vi, Val(MH_ACC_NAME)).values
    init_strategy_given_old = spl.init_strategy_constructor(
        old_raw_values, old_linked_values
    )
    # Generate some new parameters.
    # TODO(penelopeysm): This could also be an OnlyAccsVarInfo.
    new_vi = DynamicPPL.VarInfo()
    new_vi = DynamicPPL.setacc!!(new_vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    new_vi = DynamicPPL.setacc!!(new_vi, MHLinkedValuesAccumulator(spl.linkedrw_vns))
    _, new_vi = DynamicPPL.init!!(rng, model, new_vi, init_strategy_given_old)
    new_lp = DynamicPPL.getlogjoint(new_vi)
    new_raw_values = DynamicPPL.getacc(new_vi, Val(:ValuesAsInModel)).values
    new_linked_values = DynamicPPL.getacc(new_vi, Val(MH_ACC_NAME)).values
    # We need to get the priors that have been cached inside `init_strategy`.
    unspecified_priors = if init_strategy_given_old isa InitFromProposals
        init_strategy_given_old.priors
    else
        Dict{VarName,Distribution}()
    end
    init_strategy_given_new = spl.init_strategy_constructor(
        new_raw_values, new_linked_values
    )
    # Calculate the log-acceptance probability.
    @show new_raw_values new_lp
    log_a = (
        new_lp - old_lp +
        log_proposal_density(old_vi, init_strategy_given_new, unspecified_priors) -
        log_proposal_density(new_vi, init_strategy_given_old, unspecified_priors)
    )
    # Decide whether to accept.
    accepted, vi = if -Random.randexp(rng) < log_a
        true, new_vi
    else
        false, old_vi
    end
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=accepted))
    return transition, vi
end

"""
    log_proposal_density(
        old_vi::DynamicPPL.AbstractVarInfo,
        init_strategy_given_new::DynamicPPL.AbstractInitStrategy,
        unspecified_priors::Dict{VarName,Distribution}
    )

Calculate the ratio `g(x|x')` where `g` is the proposal distribution used to generate
`x` (represented by `old_vi`), given the new state `x'`.

If the arguments are switched (i.e., `new_vi` is passed as the first argument, and
`init_strategy_given_old` as the second), the function calculates `g(x'|x)`.
"""
function log_proposal_density(
    vi::DynamicPPL.AbstractVarInfo, ::DynamicPPL.InitFromPrior, ::Dict{VarName,Distribution}
)
    # Samples were drawn from the prior -- in this case g(x|x') = g(x) = prior probability
    # of x.
    return DynamicPPL.getlogprior(vi)
end
function log_proposal_density(
    vi::DynamicPPL.AbstractVarInfo,
    strategy::InitFromProposals,
    unspecified_priors::Dict{VarName,Distribution},
)
    # In this case, the proposal distribution is indeed conditional, so we need to 'run' the
    # initialisation strategies both ways. Luckily, we don't need to run the model itself,
    # since all the information we need is in the proposals. That is the reason why we have
    # to cache the priors in the InitFromProposals struct -- if any variables were not given
    # an explicit proposal (in `strategy.proposals`) we need to know what their prior was.
    vals = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
    g = 0.0
    for (vn, (is_linkedrw, proposal)) in pairs(strategy.proposals)
        if is_linkedrw
            # LinkedRW proposals end up here, but they are symmetric proposals, so we can
            # skip their contribution.
            continue
        else
            # proposal isa Distribution
            g += logpdf(proposal, vals[vn])
        end
    end
    for (vn, prior) in unspecified_priors
        g += logpdf(prior, vals[vn])
    end
    return g
end

# RWMH can be delegated to AdvancedMH.
function MH(cov_matrix::AbstractMatrix)
    return externalsampler(AdvancedMH.RWMH(MvNormal(cov_matrix)); unconstrained=true)
end

# Accumulator to store linked values; but only the ones that have a LinkedRW proposal.
const MH_ACC_NAME = :MHLinkedValuesAccumulator
struct StoreLinkedValues
    "The set of VarNames that have LinkedRW proposals."
    linkedrw_vns::Set{VarName}
end
struct MHLinkedVal{V,T}
    val::V
    sz::T
end
function (s::StoreLinkedValues)(val, tval, logjac, vn, dist)
    return if vn in s.linkedrw_vns
        linked_vec = DynamicPPL.to_linked_vec_transform(dist)(val)
        MHLinkedVal(linked_vec, size(val))
    else
        DynamicPPL.DoNotAccumulate()
    end
end
function MHLinkedValuesAccumulator(vns::Set{VarName})
    return DynamicPPL.VNTAccumulator{MH_ACC_NAME}(StoreLinkedValues(vns))
end
