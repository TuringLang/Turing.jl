function AbstractMCMC.setparams!!(
    model::DynamicPPL.Model, state::HMCState, params::AbstractVector{<:Real}
)
    new_ldf, new_params, _ = gibbs_recompute_ldf_and_params(state.ldf, model, params)
    metric = gen_metric(LogDensityProblems.dimension(new_ldf), state)
    lp_func = Base.Fix1(LogDensityProblems.logdensity, new_ldf)
    lp_grad_func = Base.Fix1(LogDensityProblems.logdensity_and_gradient, new_ldf)
    new_hamiltonian = AHMC.Hamiltonian(metric, lp_func, lp_grad_func)
    new_z = deepcopy(state.z)
    new_z.θ .= new_params
    return HMCState(state.i, state.kernel, new_hamiltonian, new_z, state.adaptor, new_ldf)
end

function AbstractMCMC.setparams!!(
    model::DynamicPPL.Model, state::TuringESSState, params::AbstractVector{<:Real}
)
    new_ldf, new_params, accs = gibbs_recompute_ldf_and_params(
        state.ldf, model, params, (DynamicPPL.LogLikelihoodAccumulator(),)
    )
    return TuringESSState(
        new_ldf, new_params, DynamicPPL.getloglikelihood(accs), state.priors
    )
end

function AbstractMCMC.setparams!!(
    model::DynamicPPL.Model,
    state::DynamicPPL.AbstractVarInfo,
    params::AbstractVector{<:Real},
)
    return last(
        DynamicPPL.init!!(
            model, state, DynamicPPL.InitFromParams(params, nothing), DynamicPPL.UnlinkAll()
        ),
    )
end

function AbstractMCMC.setparams!!(
    model::DynamicPPL.Model,
    state::DynamicPPL.AbstractVarInfo,
    params::DynamicPPL.VarNamedTuple,
)
    return last(
        DynamicPPL.init!!(
            model, state, DynamicPPL.InitFromParams(params, nothing), DynamicPPL.UnlinkAll()
        ),
    )
end

function AbstractMCMC.setparams!!(
    model::DynamicPPL.Model, state::TuringState, params::AbstractVector{<:Real}
)
    new_ldf, new_params, _ = gibbs_recompute_ldf_and_params(state.ldf, model, params)
    new_inner_state = AbstractMCMC.setparams!!(
        AbstractMCMC.LogDensityModel(new_ldf), state.state, new_params
    )
    return TuringState(new_inner_state, new_params, new_ldf)
end

function AbstractMCMC.setparams!!(
    ::DynamicPPL.Model, state::DynamicPPL.OnlyAccsVarInfo, ::AbstractVector{<:Real}
)
    return state
end

function AbstractMCMC.getparams(::DynamicPPL.Model, state::HMCState)
    return DynamicPPL.ParamsWithStats(
        state.z.θ, state.ldf; include_log_probs=false, include_colon_eq=false
    ).params
end

function AbstractMCMC.getparams(::DynamicPPL.Model, state::TuringESSState)
    return DynamicPPL.ParamsWithStats(
        state.params, state.ldf; include_log_probs=false, include_colon_eq=false
    ).params
end

function AbstractMCMC.getparams(::DynamicPPL.Model, state::TuringState)
    return DynamicPPL.ParamsWithStats(
        state.params, state.ldf; include_log_probs=false, include_colon_eq=false
    ).params
end

function AbstractMCMC.getparams(::DynamicPPL.Model, state::DynamicPPL.AbstractVarInfo)
    return gibbs_get_raw_values(state)
end
