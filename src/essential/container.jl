struct TracedModel{S<:AbstractSampler,V<:AbstractVarInfo,M<:Model,E<:Tuple}
    model::M
    sampler::S
    varinfo::V
    evaluator::E
end

function TracedModel(
    model::Model,
    sampler::AbstractSampler,
    varinfo::AbstractVarInfo,
) 
    # evaluate!!(m.model, varinfo, SamplingContext(Random.AbstractRNG, m.sampler, DefaultContext()))
    context = SamplingContext(DynamicPPL.Random.GLOBAL_RNG, sampler, DefaultContext())
    evaluator = _get_evaluator(model, varinfo, context)
    return TracedModel{AbstractSampler,AbstractVarInfo,Model,Tuple}(model, sampler, varinfo, evaluator)
end

# Smiliar to `evaluate!!` except that we return the evaluator signature without excutation.
# TODO: maybe move to DynamicPPL
@generated function _get_evaluator(
    model::Model{_F,argnames}, varinfo, context
) where {_F,argnames}
    unwrap_args = [
        :($DynamicPPL.matchingvalue(context_new, varinfo, model.args.$var)) for var in argnames
    ]
    # We want to give `context` precedence over `model.context` while also
    # preserving the leaf context of `context`. We can do this by
    # 1. Set the leaf context of `model.context` to `leafcontext(context)`.
    # 2. Set leaf context of `context` to the context resulting from (1).
    # The result is:
    # `context` -> `childcontext(context)` -> ... -> `model.context`
    #  -> `childcontext(model.context)` -> ... -> `leafcontext(context)`
    return quote
        context_new = DynamicPPL.setleafcontext(
            context, DynamicPPL.setleafcontext(model.context, DynamicPPL.leafcontext(context))
        )
        (model.f, model, DynamicPPL.resetlogp!!(varinfo), context_new, $(unwrap_args...))
    end
end

function Base.copy(trace::AdvancedPS.Trace{<:TracedModel})
    f = trace.model
    newf = TracedModel(f.model, f.sampler, deepcopy(f.varinfo))
    return AdvancedPS.Trace(newf, copy(trace.task))
end

function AdvancedPS.advance!(trace::AdvancedPS.Trace{<:AdvancedPS.GenericModel{<:TracedModel}}, isref::Bool=false)
    DynamicPPL.increment_num_produce!(trace.model.f.varinfo)
    score = consume(trace.model.ctask)
    if score === nothing
        return
    else
        return score + DynamicPPL.getlogp(trace.model.f.varinfo)
    end
end

function AdvancedPS.delete_retained!(trace::TracedModel)
    DynamicPPL.set_retained_vns_del_by_spl!(trace.varinfo, trace.sampler)
    return
end

function AdvancedPS.reset_model(trace::TracedModel)
    newvarinfo = deepcopy(trace.varinfo)
    DynamicPPL.reset_num_produce!(newvarinfo)
    return TracedModel(trace.model, trace.sampler, newvarinfo)
end

function AdvancedPS.reset_logprob!(trace::TracedModel)
    DynamicPPL.resetlogp!!(trace.model.varinfo)
    return
end

AdvancedPS.update_rng!(trace::AdvancedPS.Trace{AdvancedPS.GenericModel{TracedModel{M,S,V,E}, F}, R}) where {M,S,V,E,F,R} = nothing

function Libtask.TapedTask(model::TracedModel, ::Random.AbstractRNG)
    return Libtask.TapedTask(model.evaluator[1], model.evaluator[2:end]...)
end
