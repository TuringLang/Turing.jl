struct TracedModel{S<:AbstractSampler,V<:AbstractVarInfo,M<:Model,E<:Tuple} <:
       AdvancedPS.AbstractGenericModel
    model::M
    sampler::S
    varinfo::V
    evaluator::E
end

function TracedModel(
    model::Model,
    sampler::AbstractSampler,
    varinfo::AbstractVarInfo,
    rng::Random.AbstractRNG,
)
    context = SamplingContext(rng, sampler, DefaultContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo, context)
    if kwargs !== nothing && !isempty(kwargs)
        error(
            "Sampling with `$(sampler.alg)` does not support models with keyword arguments. See issue #2007 for more details.",
        )
    end
    return TracedModel{AbstractSampler,AbstractVarInfo,Model,Tuple}(
        model, sampler, varinfo, (model.f, args...)
    )
end

function AdvancedPS.advance!(
    trace::AdvancedPS.Trace{<:AdvancedPS.LibtaskModel{<:TracedModel}}, isref::Bool=false
)
    # Make sure we load/reset the rng in the new replaying mechanism
    DynamicPPL.increment_num_produce!(trace.model.f.varinfo)
    isref ? AdvancedPS.load_state!(trace.rng) : AdvancedPS.save_state!(trace.rng)
    score = consume(trace.model.ctask)
    if score === nothing
        return nothing
    else
        return score + DynamicPPL.getlogp(trace.model.f.varinfo)
    end
end

function AdvancedPS.delete_retained!(trace::TracedModel)
    DynamicPPL.set_retained_vns_del_by_spl!(trace.varinfo, trace.sampler)
    return trace
end

function AdvancedPS.reset_model(trace::TracedModel)
    DynamicPPL.reset_num_produce!(trace.varinfo)
    return trace
end

function AdvancedPS.reset_logprob!(trace::TracedModel)
    DynamicPPL.resetlogp!!(trace.model.varinfo)
    return trace
end

function AdvancedPS.update_rng!(
    trace::AdvancedPS.Trace{<:AdvancedPS.LibtaskModel{<:TracedModel}}
)
    # Extract the `args`.
    args = trace.model.ctask.args
    # From `args`, extract the `SamplingContext`, which contains the RNG.
    sampling_context = args[3]
    rng = sampling_context.rng
    trace.rng = rng
    return trace
end

function Libtask.TapedTask(model::TracedModel, ::Random.AbstractRNG, args...; kwargs...) # RNG ?
    return Libtask.TapedTask(model.evaluator[1], model.evaluator[2:end]...; kwargs...)
end
