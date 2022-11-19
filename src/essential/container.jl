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
    rng::Random.AbstractRNG
) 
    # evaluate!!(m.model, varinfo, SamplingContext(Random.AbstractRNG, m.sampler, DefaultContext()))
    context = SamplingContext(rng, sampler, DefaultContext())
    evaluator = _get_evaluator(model, varinfo, context)
    return TracedModel{AbstractSampler,AbstractVarInfo,Model,Tuple}(model, sampler, varinfo, evaluator)
end

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


function Base.copy(model::AdvancedPS.GenericModel{<:TracedModel})
    newtask = copy(model.ctask)
    newmodel = TracedModel{AbstractSampler,AbstractVarInfo,Model,Tuple}(deepcopy(model.f.model), deepcopy(model.f.sampler), deepcopy(model.f.varinfo), deepcopy(model.f.evaluator))
    n = AdvancedPS.GenericModel(newmodel, newtask)
    return n
end

function AdvancedPS.advance!(trace::AdvancedPS.Trace{<:AdvancedPS.GenericModel{<:TracedModel}}, isref::Bool=false)
    # Make sure we load/reset the rng in the new replaying mechanism
    DynamicPPL.increment_num_produce!(trace.model.f.varinfo)
    isref ? AdvancedPS.load_state!(trace.rng) : AdvancedPS.save_state!(trace.rng)
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
    return trace
end

function AdvancedPS.reset_logprob!(trace::TracedModel)
    DynamicPPL.resetlogp!!(trace.model.varinfo)
    return
end

function AdvancedPS.update_rng!(trace::AdvancedPS.Trace{AdvancedPS.GenericModel{TracedModel{M,S,V,E}, F}, R}) where {M,S,V,E,F,R} 
    args = trace.model.ctask.args
    _, _, container, = args
    rng = container.rng
    trace.rng = rng
end

function Libtask.TapedTask(model::TracedModel, rng::Random.AbstractRNG; kwargs...)
    return Libtask.TapedTask(model.evaluator[1], model.evaluator[2:end]...; kwargs...)
end
