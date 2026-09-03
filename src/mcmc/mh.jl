"""
    MH()

Construct a Metropolis-Hastings sampler that draws proposals from the model prior.

    MH(cov_matrix)

Construct a Metropolis-Hastings sampler that performs random-walk sampling in linked
space, with proposals drawn from a multivariate normal distribution with the given
covariance matrix. Its dimension must match the complete linked parameter vector.
"""
struct MH <: AbstractSampler end

function MH(::Pair, ::Vararg{Pair})
    throw(
        ArgumentError(
            "Per-variable MH proposals are not supported. Use `MH()` for prior " *
            "proposals or `MH(cov_matrix)` for a random walk over the complete linked " *
            "parameter vector.",
        ),
    )
end

"""
    MHState(vi, accepted)

`MH`'s state: the varinfo it returns, and whether its last step accepted.

The flag rides on the transition as well, but `Gibbs` steps its components with
`discard_sample=true` and keeps only the state, so a statistic reachable only from the
transition never reaches the chain.
"""
struct MHState{V<:DynamicPPL.AbstractVarInfo}
    vi::V
    accepted::Bool
end

function mh_varinfo()
    return DynamicPPL.setacc!!(
        DynamicPPL.OnlyAccsVarInfo(), DynamicPPL.RawValueAccumulator(true)
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
    vi = mh_varinfo()
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params, DynamicPPL.UnlinkAll())

    initial_logprior = DynamicPPL.getlogprior(vi)
    if initial_logprior == -Inf || isnan(initial_logprior)
        io = IOContext(IOBuffer(), :color => true)
        show(io, "text/plain", DynamicPPL.get_raw_values(vi))
        init_str = String(take!(io.io))
        density_description = initial_logprior == -Inf ? "zero" : "a NaN"
        error(
            "The initial parameters have $density_description probability density under" *
            " the model prior, which MH uses as its proposal distribution. This will" *
            " cause the sampler to get stuck at the initial parameters. Consider" *
            " specifying different initial parameters (e.g. via `InitFromParams`)." *
            " Your initial values were:\n\n$init_str\n",
        )
    end

    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=true))
    return transition, MHState(vi, true)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::MH,
    state::MHState;
    discard_sample=false,
    kwargs...,
)
    old_vi = state.vi
    old_lp = DynamicPPL.getlogjoint_internal(old_vi)

    new_vi = mh_varinfo()
    _, new_vi = DynamicPPL.init!!(
        rng, model, new_vi, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
    )
    new_lp = DynamicPPL.getlogjoint_internal(new_vi)

    log_a =
        new_lp - old_lp + DynamicPPL.getlogprior(old_vi) - DynamicPPL.getlogprior(new_vi)
    isnan(log_a) && @warn "MH log-acceptance probability is NaN; sample will be rejected"

    accepted, vi = if -Random.randexp(rng) < log_a
        true, new_vi
    else
        false, old_vi
    end
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=accepted))
    return transition, MHState(vi, accepted)
end

# RWMH can be delegated to AdvancedMH. The type bound is intentionally lax because we just
# let the MvNormal constructor handle it.
function MH(cov_matrix::Any)
    return externalsampler(AMH.RWMH(MvNormal(cov_matrix)); unconstrained=true)
end

####
#### Gibbs interface
####

# `gibbs_get_parameter_values` is deliberately not specialised -- gibbs.jl keeps it as the sole
# entry point so the deprecated name is still consulted -- so the values come through here.
_default_parameter_values(state::MHState) = DynamicPPL.get_parameter_values(state.vi)
gibbs_get_stats(state::MHState) = (; accepted=state.accepted)

function gibbs_update_state!!(
    ::MH, state::MHState, model::DynamicPPL.Model, global_vals::DynamicPPL.VarNamedTuple
)
    # Reevaluate the model to reflect values changed by the other Gibbs components.
    init_strat = DynamicPPL.InitFromParams(global_vals, nothing)
    vi = last(DynamicPPL.init!!(model, state.vi, init_strat, DynamicPPL.UnlinkAll()))
    return MHState(vi, state.accepted)
end

# `MH` reproposes from the values it is handed and keeps no layout, so a block another
# component reshaped needs nothing special.
function gibbs_update_state!!(
    spl::MH,
    state::MHState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
    ::ReshapedBlock,
)
    return gibbs_update_state!!(spl, state, model, global_vals)
end
