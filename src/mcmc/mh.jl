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
# Use a static proposal for s² (which happens to be the same
# as the prior) and a static proposal for m (note that this
# isn't a random walk proposal).
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

Providing a covariance matrix will cause `MH` to perform random-walk sampling in the
transformed space with proposals drawn from a multivariate normal distribution. The provided
matrix must be positive semi-definite and square:

```julia
# Providing a custom variance-covariance matrix
spl = MH(
    [0.25 0.05;
     0.05 0.50]
)
```
"""
struct MH{I} <: AbstractSampler
    "A function which takes the VarNamedTuple of values at the previous step and returns an
    AbstractInitStrategy. We don't have access to the VNT until the actual sampling, so we
    have to use a function here."
    init_strategy_constructor::I
end
MH() = MH(vnt -> DynamicPPL.InitFromPrior())

struct InitFromProposals{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    "A mapping of VarNames to Distributions that they should be sampled from. If the VarName
    is not in this VarNamedTuple, then it will be sampled from the prior."
    proposals::V
    priors::Dict{VarName,Distribution}
end
function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, prior::Distribution, strategy::InitFromProposals
)
    dist = if haskey(strategy.proposals, vn)
        # this is the proposal that the user wanted
        strategy.proposals[vn]
    else
        # dist is the prior. We need to cache it for later use when calculating the proposal
        # density.
        strategy.priors[vn] = prior
        prior
    end
    return DynamicPPL.UntransformedValue(rand(rng, dist))
end

const SymOrVNPair = Pair{<:Union{Symbol,VarName},<:Any}

function MH(pair1::SymOrVNPair, pairs::Vararg{SymOrVNPair})
    vn_proposal_pairs = (pair1, pairs...)
    return MH(
        # It is assumed that `vnt` is a VarNamedTuple that has all the variables' values
        # already set. NOTE: It doesn't store `AbstractTransformedValue`s, but the actual
        # raw values.
        vnt -> begin
            proposals = DynamicPPL.VarNamedTuple()
            for pair in vn_proposal_pairs
                vn, proposal = pair
                # Convert all keys to VarNames.
                if vn isa Symbol
                    vn = DynamicPPL.VarName{vn}()
                elseif !(vn isa DynamicPPL.VarName)
                    throw(
                        ArgumentError(
                            "first element of each pair must be a Symbol or VarName"
                        ),
                    )
                end
                # Check whether the proposal is a Distribution.
                proposal_dist = if proposal isa Distribution
                    proposal
                else
                    # It's a callable that takes `vnt` and returns a distribution.
                    proposal(vnt)
                end
                proposals = DynamicPPL.templated_setindex!!(
                    proposals, proposal_dist, vn, vnt.data[AbstractPPL.getsym(vn)]
                )
            end
            return InitFromProposals(proposals, Dict{VarName,Distribution}())
        end,
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::MH;
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
    old_params = DynamicPPL.getacc(old_vi, Val(:ValuesAsInModel)).values
    init_strategy_given_old = spl.init_strategy_constructor(old_params)
    # Generate some new parameters.
    # TODO(penelopeysm): This could also be an OnlyAccsVarInfo.
    new_vi = DynamicPPL.VarInfo()
    new_vi = DynamicPPL.setacc!!(new_vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    _, new_vi = DynamicPPL.init!!(rng, model, new_vi, init_strategy_given_old)
    new_lp = DynamicPPL.getlogjoint(new_vi)
    new_params = DynamicPPL.getacc(new_vi, Val(:ValuesAsInModel)).values
    # We need to get the priors that have been cached inside `init_strategy`.
    unspecified_priors = if init_strategy_given_old isa InitFromProposals
        init_strategy_given_old.priors
    else
        Dict{VarName,Distribution}()
    end
    init_strategy_given_new = spl.init_strategy_constructor(new_params)
    # Calculate the log-acceptance probability.
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
    for (vn, proposal) in pairs(strategy.proposals)
        # proposal isa Distribution
        g += logpdf(proposal, vals[vn])
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
