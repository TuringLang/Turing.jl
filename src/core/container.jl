struct TracedModel{S<:AbstractSampler,V<:AbstractVarInfo,M<:Model}
    model::M
    sampler::S
    varinfo::V
end

# needed?
function TracedModel{SampleFromPrior}(
    model::Model,
    sampler::AbstractSampler,
    varinfo::AbstractVarInfo,
)
    return TracedModel(model, SampleFromPrior(), varinfo)
end

(f::TracedModel)() = f.model(f.varinfo, f.sampler)

function Base.copy(trace::AdvancedPS.Trace{<:TracedModel})
    f = trace.f
    newf = TracedModel(f.model, f.sampler, deepcopy(f.varinfo))
    return AdvancedPS.Trace(newf, copy(trace.ctask))
end

function AdvancedPS.advance!(trace::AdvancedPS.Trace{<:TracedModel})
    DynamicPPL.increment_num_produce!(trace.f.varinfo)
    score = consume(trace.ctask)
    if score === nothing
        return
    else
        return score + DynamicPPL.getlogp(trace.f.varinfo)
    end
end

function AdvancedPS.delete_retained!(f::TracedModel)
    DynamicPPL.set_retained_vns_del_by_spl!(f.varinfo, f.sampler)
    return
end

function AdvancedPS.reset_model(f::TracedModel)
    newvarinfo = deepcopy(f.varinfo)
    DynamicPPL.reset_num_produce!(newvarinfo)
    return TracedModel(f.model, f.sampler, newvarinfo)
end

function AdvancedPS.reset_logprob!(f::TracedModel)
    DynamicPPL.resetlogp!(f.varinfo)
    return
end

