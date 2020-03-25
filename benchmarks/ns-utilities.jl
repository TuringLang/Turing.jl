
using DataFrames

## copied from NanoSoldier.jl
# from submisson.jl
# `x` can only be Expr, Symbol, QuoteNode, T<:Number, or T<:AbstractString
phrase_argument(x::Union{Expr, Symbol, QuoteNode}) = string(x)
phrase_argument(x::Union{AbstractString, Number})  = repr(x)

function parse_submission_string(submission_string)
    fncall = match(r"`.*?`", submission_string).match[2:end-1]
    argind = findfirst(isequal('('), fncall)
    name = fncall[1:(argind - 1)]
    parsed_args = Meta.parse(replace(fncall[argind:end], ";" => ","))
    args, kwargs = Vector{String}(), Dict{Symbol,String}()
    if isa(parsed_args, Expr) && parsed_args.head == :tuple
        started_kwargs = false
        for x in parsed_args.args
            if isa(x, Expr) && (x.head == :kw || x.head == :(=)) && isa(x.args[1], Symbol)
                @assert !haskey(kwargs, x.args[1]) "kwargs must all be unique"
                kwargs[x.args[1]] = phrase_argument(x.args[2])
                started_kwargs = true
            else
                @assert !started_kwargs "kwargs must come after other args"
                push!(args, phrase_argument(x))
            end
        end
    else
        push!(args, phrase_argument(parsed_args))
    end
    return name, args, kwargs
end


## format benchmark results


function Base.convert(::Type{Dict}, t::BenchmarkTools.Trial)
    data = Dict{String, Float64}()
    data["times"] = sum(t.times) / length(t.times)
    data["gctimes"] = sum(t.gctimes) / length(t.gctimes)
    data["memory"] = t.memory
    data["allocs"] = t.allocs
    return data
end

function flatten_benchmark_results(res, data::Dict{String, Dict}, prefix = "")
    for key in keys(res)
        if res[key] isa BenchmarkTools.Trial
            data["$prefix.$key"] = convert(Dict, res[key])
        elseif res[key] isa BenchmarkGroup
            flatten_benchmark_results(res[key], data, "$prefix.$key")
        end
    end
end

function bmresults_to_dataframe(res)
    flatten_res = Dict{String, Dict}()
    flatten_benchmark_results(res, flatten_res)

    names = collect(keys(flatten_res))
    df= DataFrame(Name=names,
                  Times=map(x->flatten_res[x]["times"], names),
                  GCtimes=map(x->flatten_res[x]["gctimes"], names),
                  Memory=map(x->flatten_res[x]["memory"], names),
                  Allocs=map(x->flatten_res[x]["allocs"], names),
                  )
    return df
end
