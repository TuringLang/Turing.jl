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
    return transition, vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::MH,
    old_vi::DynamicPPL.OnlyAccsVarInfo;
    discard_sample=false,
    kwargs...,
)
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
    return transition, vi
end

# RWMH can be delegated to AdvancedMH. The type bound is intentionally lax because we just
# let the MvNormal constructor handle it.
function MH(cov_matrix::Any)
    return externalsampler(AMH.RWMH(MvNormal(cov_matrix)); unconstrained=true)
end

####
#### Gibbs interface
####

function gibbs_update_state!!(
    ::MH,
    state::AbstractVarInfo,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    # Reevaluate the model to reflect values changed by the other Gibbs components.
    init_strat = DynamicPPL.InitFromParams(global_vals, nothing)
    return last(DynamicPPL.init!!(model, state, init_strat, DynamicPPL.UnlinkAll()))
end
