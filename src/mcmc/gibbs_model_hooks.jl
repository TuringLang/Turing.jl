# Disambiguate: (DynamicPPL.Model, AbstractMCMC.Gibbs) is more specific than
# both (AbstractModel, Gibbs) and (DynamicPPL.Model, AbstractSampler).
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::AbstractMCMC.Gibbs;
    initial_params=nothing,
    kwargs...,
)
    # Turing passes InitFromPrior/InitFromParams as initial_params.
    # Treat anything that is not a VarNamedTuple as "no prior values" for condition().
    # Still forward initial_params to component samplers so they can initialise.
    gv = initial_params isa DynamicPPL.VarNamedTuple ? initial_params : nothing
    component_initial_params =
        initial_params === nothing ? DynamicPPL.InitFromPrior() : initial_params
    sub_states = AbstractMCMC._gibbs_initial_steps(
        rng,
        model,
        spl.varnames,
        spl.samplers,
        gv;
        initial_params=component_initial_params,
        kwargs...,
    )
    global_values = AbstractMCMC._collect_global_values(
        model, spl.varnames, spl.samplers, sub_states
    )
    return AbstractMCMC._build_gibbs_transition(global_values),
    AbstractMCMC.GibbsState(global_values, sub_states)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::AbstractMCMC.Gibbs,
    state::AbstractMCMC.GibbsState;
    kwargs...,
)
    global_values, sub_states = AbstractMCMC._gibbs_sweep(
        rng,
        model,
        spl.varnames,
        spl.samplers,
        state.sub_states,
        state.global_values;
        kwargs...,
    )
    return AbstractMCMC._build_gibbs_transition(global_values),
    AbstractMCMC.GibbsState(global_values, sub_states)
end

function AbstractMCMC.condition(
    model::DynamicPPL.Model,
    target_varnames::AbstractVector{<:VarName},
    global_values::DynamicPPL.VarNamedTuple,
)
    conditioned_model, _ctx = make_conditional(model, target_varnames, global_values)
    return conditioned_model
end

function AbstractMCMC.condition(
    model::DynamicPPL.Model, target_varnames::AbstractVector{<:VarName}, ::Nothing
)
    return model
end

function AbstractMCMC._init_global_values(
    ::DynamicPPL.Model, ::AbstractVector{<:VarName}, ::DynamicPPL.Model, sub_state
)
    return gibbs_get_raw_values(sub_state)
end

function AbstractMCMC._update_global_values(
    ::DynamicPPL.Model,
    global_values::DynamicPPL.VarNamedTuple,
    ::AbstractVector{<:VarName},
    cond_model::DynamicPPL.Model,
    new_params::AbstractVector{<:Real},
)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(
        cond_model,
        accs,
        DynamicPPL.InitFromParams(new_params, nothing),
        DynamicPPL.UnlinkAll(),
    )
    return merge(global_values, DynamicPPL.get_raw_values(accs))
end

# VarNamedTuple overload: MH's getparams returns a VarNamedTuple directly;
# merge it straight into global_values without the encode/decode roundtrip.
function AbstractMCMC._update_global_values(
    ::DynamicPPL.Model,
    global_values::DynamicPPL.VarNamedTuple,
    ::AbstractVector{<:VarName},
    ::DynamicPPL.Model,
    new_params::DynamicPPL.VarNamedTuple,
)
    return merge(global_values, new_params)
end

function AbstractMCMC._build_gibbs_transition(global_values::DynamicPPL.VarNamedTuple)
    return global_values
end
