mutable struct Trace{Tspl<:AbstractSampler, Tvi<:AbstractVarInfo, Tmodel<:Model}
    model::Tmodel
    spl::Tspl
    vi::Tvi
    ctask::CTask

    function Trace{SampleFromPrior}(model::Model, spl::AbstractSampler, vi::AbstractVarInfo)
        return new{SampleFromPrior,typeof(vi),typeof(model)}(model, SampleFromPrior(), vi)
    end
    function Trace{S}(model::Model, spl::S, vi::AbstractVarInfo) where S<:Sampler
        return new{S,typeof(vi),typeof(model)}(model, spl, vi)
    end
end

function Base.copy(trace::Trace)
    vi = deepcopy(trace.vi)
    res = Trace{typeof(trace.spl)}(trace.model, trace.spl, vi)
    res.ctask = copy(trace.ctask)
    return res
end

# NOTE: this function is called by `forkr`
function Trace(f, m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi))
    ctask = CTask() do
        res = f()
        produce(nothing)
        return res
    end
    task = ctask.task
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    res.ctask = ctask
    return res
end

function Trace(m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi))
    reset_num_produce!(res.vi)
    ctask = CTask() do
        res = m(vi, spl)
        produce(nothing)
        return res
    end
    task = ctask.task
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    res.ctask = ctask
    return res
end

# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (increment_num_produce!(t.vi); consume(t.ctask))

# Task copying version of fork for Trace.
function fork(trace :: Trace, is_ref :: Bool = false)
    newtrace = copy(trace)
    is_ref && set_retained_vns_del_by_spl!(newtrace.vi, newtrace.spl)
    newtrace.ctask.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace::Trace)
    newtrace = Trace(trace.ctask.task.code, trace.model, trace.spl, deepcopy(trace.vi))
    newtrace.spl = trace.spl
    reset_num_produce!(newtrace.vi)
    return newtrace
end

current_trace() = current_task().storage[:turing_trace]

const Particle = Trace