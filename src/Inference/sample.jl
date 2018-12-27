##############
# Utilities  #
##############

# VarInfo to Sample
function Sample(vi::UntypedVarInfo)
    value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
    for vn in keys(vi)
        value[sym(vn)] = vi[vn]
    end

    # NOTE: do we need to check if lp is 0?
    value[:lp] = getlogp(vi)
    Sample(0.0, value)
end
@generated function Sample(vi::TypedVarInfo{Tvis}) where Tvis
    expr = quote
        value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
    end
    for f in fieldnames(Tvis)
        push!(expr.args, quote
            for vn in keys(vi.vis.$f.idcs)
                value[sym(vn)] = vi.vis.$f.idcs[vn]
            end
        end)
    end
    push!(expr.args, quote
        # NOTE: do we need to check if lp is 0?
        value[:lp] = getlogp(vi)
        Sample(0.0, value)
    end)
    return expr
end

# VarInfo, combined with spl.info, to Sample
function Sample(vi::AbstractVarInfo, spl::Sampler)
    s = Sample(vi)

    if haskey(spl.info, :wum)
        s.value[:epsilon] = getss(spl.info[:wum])
    end

    if haskey(spl.info, :lf_num)
        s.value[:lf_num] = spl.info[:lf_num]
    end

    if haskey(spl.info, :eval_num)
        s.value[:eval_num] = spl.info[:eval_num]
    end

    return s
end

function sample(model, alg; kwargs...)
    spl = get_sampler(model, alg; kwargs...)
    samples = init_samples(alg; kwargs...)
    vi = init_varinfo(model, spl; kwargs...)
    _sample(vi, samples, spl, model, alg; kwargs...)
end

function init_samples(alg; kwargs...)
    n = get_sample_n(alg; kwargs...)
    weight = 1 / n
    samples = init_samples(n, weight)
    return samples
end

function get_sample_n(alg; reuse_spl_n = 0, kwargs...)
    if reuse_spl_n > 0
        return reuse_spl_n
    else
        alg.n_iters
    end
end

function init_samples(sample_n, weight)
    samples = Array{Sample}(undef, sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    return samples
end

function get_sampler(model, alg; kwargs...)
    spl = default_sampler(model, alg; kwargs...)
    if alg isa AbstractGibbs
        @assert typeof(spl.alg) == typeof(alg) "[Turing] alg type mismatch; please use resume() to re-use spl"
    end
    return spl
end

function default_sampler(model, alg; reuse_spl_n = 0, resume_from = nothing)
    if reuse_spl_n > 0
        return resume_from.info[:spl]
    else
        return Sampler(alg, model)
    end
end

function init_varinfo(model, spl; kwargs...)
    vi = TypedVarInfo(default_varinfo(model, spl; kwargs...))
    return vi
end

function default_varinfo(model, spl; resume_from = nothing, kwargs...)
    if resume_from == nothing
        vi = VarInfo()
        model(vi, HamiltonianRobustInit())
        return vi
    else
        return resume_from.info[:vi]
    end
end
