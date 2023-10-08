function unique_tuple(xs::Tuple, acc::Tuple = ())
    return if Base.first(xs) ∈ acc
        unique_tuple(Base.tail(xs), acc)
    else
        unique_tuple(Base.tail(xs), (acc..., Base.first(xs)))
    end
end
unique_tuple(::Tuple{}, acc::Tuple = ()) = acc

subset(vi::DynamicPPL.TypedVarInfo, vns::Union{Tuple,AbstractArray}) = subset(vi, vns...)
function subset(vi::DynamicPPL.TypedVarInfo, vns::VarName...)
    # TODO: peform proper check of the meatdatas corresponding to different symbols.
    # F. ex. we might have vns `(@varname(x[1]), @varname(x[2]))`, in which case they
    # have the same `metadata`. If they don't, we should error.

    # TODO: Handle mixing of symbols, e.g. `(@varname(x[1]), @varname(y[1]))`.
    vns_unique_syms = unique_tuple(map(DynamicPPL.getsym, vns))
    mds = map(Base.Fix1(DynamicPPL.getfield, vi.metadata), vns_unique_syms)
    return DynamicPPL.VarInfo(NamedTuple{vns_unique_syms}(mds), vi.logp, vi.num_produce)
end

subset(vi::DynamicPPL.SimpleVarInfo, vns::Union{Tuple,AbstractArray}) = subset(vi, vns...)
function subset(vi::DynamicPPL.SimpleVarInfo, vns::VarName...)
    vals = map(Base.Fix1(getindex, vi), vns)
    return DynamicPPL.BangBang.@set!! vi.values = vals
end

function merge_metadata(md::DynamicPPL.Metadata, md_subset::DynamicPPL.Metadata)
    @assert md.vns == md_subset.vns "Cannot merge metadata with different vns."
    @assert length(md.vals) == length(md_subset.vals) "Cannot merge metadata with different length vals."

    # TODO: Re-adjust `ranges`, etc. so we can support things like changing support, etc.
    return DynamicPPL.Metadata(
        md_subset.idcs,
        md_subset.vns,
        md_subset.ranges,
        md_subset.vals,
        md_subset.dists,
        md_subset.gids,
        md_subset.orders,
        md_subset.flags,
    )
end

function merge_varinfo(
    vi::DynamicPPL.VarInfo{<:NamedTuple{names}},
    vi_subset::TypedVarInfo,
) where {names}
    # Assumes `vi` is a superset of `vi_subset`.
    metadata_vals = map(names) do vn_sym
        # TODO: Make generated.
        return if haskey(vi_subset, VarName{vn_sym}())
            merge_metadata(vi.metadata[vn_sym], vi_subset.metadata[vn_sym])
        else
            vi.metadata[vn_sym]
        end
    end

    # TODO: Is this the right way to do this?
    return DynamicPPL.VarInfo(NamedTuple{names}(metadata_vals), vi.logp, vi.num_produce)
end

function merge_varinfo(vi_left::SimpleVarInfo, vi_right::SimpleVarInfo)
    return SimpleVarInfo(
        merge(vi_left.values, vi_right.values),
        vi_left.logp + vi_right.logp,
    )
end

function merge_varinfo(vi_left::TypedVarInfo, vi_right::TypedVarInfo)
    return TypedVarInfo(
        merge_metadata(vi_left.metadata, vi_right.metadata),
        vi_left.logp + vi_right.logp,
    )
end

# TODO: Move to DynamicPPL.
DynamicPPL.condition(model::Model, varinfo::SimpleVarInfo) =
    DynamicPPL.condition(model, DynamicPPL.values_as(varinfo))
function DynamicPPL.condition(model::Model, varinfo::VarInfo)
    # Use `OrderedDict` as default for `VarInfo`.
    # TODO: Do better!
    return DynamicPPL.condition(model, DynamicPPL.values_as(varinfo, OrderedDict))
end

# Recursive definition.
function DynamicPPL.condition(model::Model, varinfos::AbstractVarInfo...)
    return DynamicPPL.condition(
        DynamicPPL.condition(model, first(varinfos)),
        Base.tail(varinfos)...,
    )
end
DynamicPPL.condition(model::Model, ::Tuple{}) = model


"""
    make_conditional_model(model, varinfo, varinfos)

Construct a conditional model from `model` conditioned `varinfos`, excluding `varinfo` if present.

# Examples
```julia-repl
julia> model = DynamicPPL.TestUtils.demo_assume_dot_observe();

julia> # A separate varinfo for each variable in `model`.
       varinfos = (DynamicPPL.SimpleVarInfo(s=1.0), DynamicPPL.SimpleVarInfo(m=10.0));

julia> # The varinfo we want to NOT condition on.
       target_varinfo = first(varinfos);

julia> # Results in a model with only `m` conditioned.
       conditioned_model = Turing.Inference.make_conditional(model, target_varinfo, varinfos);

julia> result = conditioned_model();

julia> result.m == 10.0  # we conditioned on varinfo with `m = 10.0`
true

julia> result.s != 1.0  # we did NOT want to condition on varinfo with `s = 1.0`
true
```
"""
function make_conditional(model::Model, target_varinfo::AbstractVarInfo, varinfos)
    # TODO: Check if this is known at compile-time if `varinfos isa Tuple`.
    return DynamicPPL.condition(
        model,
        filter(Base.Fix1(!==, target_varinfo), varinfos)...
    )
end

wrap_algorithm_maybe(x) = x
wrap_algorithm_maybe(x::InferenceAlgorithm) = Sampler(x)

struct GibbsV2{V,A} <: InferenceAlgorithm
    varnames::V
    samplers::A
end

# NamedTuple
GibbsV2(; algs...) = GibbsV2(NamedTuple(algs))
function GibbsV2(algs::NamedTuple)
    return GibbsV2(
        map(s -> VarName{s}(), keys(algs)),
        map(wrap_algorithm_maybe, values(algs)),
    )
end

# AbstractDict
function GibbsV2(algs::AbstractDict)
    return GibbsV2(keys(algs), map(wrap_algorithm_maybe, values(algs)))
end
function GibbsV2(algs::Pair...)
    return GibbsV2(map(first, algs), map(wrap_algorithm_maybe, map(last, algs)))
end
GibbsV2(algs::Tuple) = GibbsV2(Dict(algs))

struct GibbsV2State{V<:AbstractVarInfo,S}
    vi::V
    states::S
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsV2};
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers

    # 1. Run the model once to get the varnames present + initial values to condition on.
    vi_base = DynamicPPL.VarInfo(model)
    varinfos = map(Base.Fix1(subset, vi_base), varnames)

    # 2. Construct a varinfo for every vn + sampler combo.
    states_and_varinfos = map(samplers, varinfos) do sampler_local, varinfo_local
        # Construct the conditional model.
        model_local = make_conditional(model, varinfo_local, varinfos)

        # Take initial step.
        new_state_local = last(AbstractMCMC.step(rng, model_local, sampler_local; kwargs...))

        # Return the new state and the invlinked `varinfo`.
        vi_local_state = varinfo(new_state_local)
        vi_local_state_linked = if DynamicPPL.istrans(vi_local_state)
            DynamicPPL.invlink(vi_local_state, sampler_local, model_local)
        else
            vi_local_state
        end
        return (new_state_local, vi_local_state_linked)
    end

    states = map(first, states_and_varinfos)
    varinfos = map(last, states_and_varinfos)

    # Update the base varinfo from the first varinfo and replace it.
    varinfos_new = DynamicPPL.setindex!!(varinfos, vi_base, 1)
    # Merge the updated initial varinfo with the rest of the varinfos + update the logp.
    vi = DynamicPPL.setlogp!!(
        reduce(merge_varinfo, varinfos_new),
        DynamicPPL.getlogp(last(varinfos)),
    )

    return Transition(model, vi), GibbsV2State(vi, states)
end

function gibbs_step_inner(
    rng::Random.AbstractRNG,
    model::Model,
    samplers,
    states,
    varinfos,
    index;
    kwargs...,
)
    # Needs to do a a few things.
    sampler_local = samplers[index]
    state_local = states[index]
    varinfo_local = varinfos[index]

    # 1. Create conditional model.
    # Construct the conditional model.
    # NOTE: Here it's crucial that all the `varinfos` are in the constrained space,
    # otherwise we're conditioning on values which are not in the support of the
    # distributions.
    model_local = make_conditional(model, varinfo_local, varinfos)

    # TODO: Might need to re-run the model.
    # NOTE: We use `logjoint` instead of `evaluate!!` and capturing the resulting varinfo because
    # the resulting varinfo might be in un-transformed space even if `varinfo_local`
    # is in transformed space. This can occur if we hit `maybe_invlink_before_eval!!`.
    varinfo_local = DynamicPPL.setlogp!!(
        varinfo_local,
        DynamicPPL.logjoint(model_local, varinfo_local),
    )

    # 2. Take step with local sampler.
    # Update the state we're about to use if need be.
    # If the sampler requires a linked varinfo, this should be done in `gibbs_state`.
    current_state_local = gibbs_state(
        model_local, sampler_local, state_local, varinfo_local
    )

    # Take a step.
    new_state_local = last(
        AbstractMCMC.step(
            rng,
            model_local,
            sampler_local,
            current_state_local;
            kwargs...,
        ),
    )

    # 3. Extract the new varinfo.
    # Return the resulting state and invlinked `varinfo`.
    varinfo_local_state = varinfo(new_state_local)
    varinfo_local_state_invlinked = if DynamicPPL.istrans(varinfo_local_state)
        DynamicPPL.invlink(varinfo_local_state, sampler_local, model_local)
    else
        varinfo_local_state
    end

    # TODO: alternatively, we can return `states_new, varinfos_new, index_new`
    return (new_state_local, varinfo_local_state_invlinked)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsV2},
    state::GibbsV2State;
    kwargs...,
)
    alg = spl.alg
    samplers = alg.samplers
    states = state.states
    varinfos = map(varinfo, state.states)
    @assert length(samplers) == length(state.states)

    # TODO: move this into a recursive function so we can unroll when reasonable?
    for index = 1:length(samplers)
        # Take the inner step.
        new_state_local, new_varinfo_local = gibbs_step_inner(
            rng,
            model,
            samplers,
            states,
            varinfos,
            index;
            kwargs...,
        )

        # Update the `states` and `varinfos`.
        states = Setfield.setindex(states, new_state_local, index)
        varinfos = Setfield.setindex(varinfos, new_varinfo_local, index)
    end

    # Combine the resulting varinfo objects.
    # The last varinfo holds the correctly computed logp.
    vi_base = state.vi

    # Update the base varinfo from the first varinfo and replace it.
    varinfos_new = DynamicPPL.setindex!!(
        varinfos,
        merge_varinfo(vi_base, first(varinfos)),
        firstindex(varinfos),
    )
    # Merge the updated initial varinfo with the rest of the varinfos + update the logp.
    vi = DynamicPPL.setlogp!!(
        reduce(merge_varinfo, varinfos_new),
        DynamicPPL.getlogp(last(varinfos)),
    )

    return Transition(model, vi), GibbsV2State(vi, states)
end
