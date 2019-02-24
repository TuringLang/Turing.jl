#########################
# Sampler I/O Interface #
#########################

##########
# Sample #
##########

mutable struct Sample
    weight :: Float64     # particle weight
    value :: Dict{Symbol,Any}
end

Base.getindex(s::Sample, v::Symbol) = getjuliatype(s, v)

function parse_inds(inds)
    p_inds = [parse(Int, m.captures[1]) for m in eachmatch(r"(\d+)", inds)]
    if length(p_inds) == 1
        return p_inds[1]
    else
        return Tuple(p_inds)
    end
end

function getjuliatype(s::Sample, v::Symbol, cached_syms=nothing)
    # NOTE: cached_syms is used to cache the filter entiries in svalue. This is helpful when the dimension of model is huge.
    if cached_syms == nothing
        # Get all keys associated with the given symbol
        syms = collect(Iterators.filter(k -> occursin(string(v)*"[", string(k)), keys(s.value)))
    else
        syms = collect((Iterators.filter(k -> occursin(string(v), string(k)), cached_syms)))
    end

    # Map to the corresponding indices part
    idx_str = map(sym -> replace(string(sym), string(v) => ""), syms)
    # Get the indexing component
    idx_comp = map(idx -> collect(Iterators.filter(str -> str != "", split(string(idx), [']','[']))), idx_str)

    # Deal with v is really a symbol, e.g. :x
    if isempty(idx_comp)
        @assert haskey(s.value, v)
        return Base.getindex(s.value, v)
    end

    # Construct container for the frist nesting layer
    dim = length(split(idx_comp[1][1], ','))
    if dim == 1
        sample = Vector(undef, length(unique(map(c -> c[1], idx_comp))))
    else
        d = max(map(c -> parse_inds(c[1]), idx_comp)...)
        sample = Array{Any, length(d)}(undef, d)
    end

    # Fill sample
    for i = 1:length(syms)
        # Get indexing
        idx = parse_inds(idx_comp[i][1])
        # Determine if nesting
        nested_dim = length(idx_comp[1]) # how many nested layers?
        if nested_dim == 1
            setindex!(sample, getindex(s.value, syms[i]), idx...)
        else  # nested case, iteratively evaluation
            v_indexed = Symbol("$v[$(idx_comp[i][1])]")
            setindex!(sample, getjuliatype(s, v_indexed, syms), idx...)
        end
    end
    return sample
end

#########
# Chain #
#########

# Variables to put in the Chains :internal section.
const _internal_vars = ["elapsed",
 "epsilon",
 "eval_num",
 "lf_eps",
 "lf_num",
 "lp"]


function Chain(w::Real, s::AbstractArray{Sample})
    samples = flatten.(s)
    names_ = collect(mapreduce(s -> keys(s), union, samples))

    values_ = mapreduce(v -> map(k -> haskey(v, k) ? v[k] : missing, names_), hcat, samples)
    values_ = convert(Array{Union{Missing, Float64}, 2}, values_')

    chn = Chains(
        reshape(values_, size(values_, 1), size(values_, 2), 1),
        names_,
        Dict(:internals => _internal_vars),
        evidence = w
    )
    # chn = Chain(
    #             w,
    #             c.value,
    #             c.range,
    #             c.names,
    #             c.chains,
    #             Dict{Symbol, Any}()
    #            )
    return chn
end

# ind2sub is deprecated in Julia 1.0
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

function flatten(s::Sample)
    vals  = Vector{Float64}()
    names = Vector{AbstractString}()
    for (k, v) in s.value
        flatten(names, vals, string(k), v)
    end
    return Dict(names[i] => vals[i] for i in 1:length(vals))
end

function flatten(names, value :: Array{Float64}, k :: String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = k * string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, Float64(v[i]))
                push!(names, name)
            elseif isa(v[i], Array)
                name = k * string(ind2sub(size(v), i))
                flatten(names, value, name, v[i])
            else
                error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
            end
        end
    else
        error("Unknown var type: typeof($v)=$(typeof(v))")
    end
    return
end

function Base.getindex(c::Chains, v::Symbol)
    # This strange implementation is mostly to keep backward compatability.
    #  Needs some refactoring a better format for storing results is available.
    if v == :logevidence
        return c.logevidence
    elseif v==:logweights
        return c["lp"]
    else
        idx = indexin(string(v), names(c))
        if any(idx .== nothing)
            syms = collect(Iterators.filter(k -> occursin(string(v)*"[", string(k)), names(c)))
            sort!(syms)
            return c.value[:, syms, :]
        else
            return c[string(v)]
        end
    end
end

# function Base.getindex(c::Chain, v::String)
#     return c.value[:, names2inds(c, v), :]
# end


# function Base.getindex(c::Chains, expr::Expr)
#     str = replace(string(expr), r"\(|\)" => "")
#     @assert match(r"^\w+(\[(\d\,?)*\])+$", str) != nothing "[Turing.jl] $expr invalid for getindex(::Chain, ::Expr)"
#     return c[Symbol(str)]
# end

function Base.vcat(c1::Chains, args::Chains...)
    names = names(c1)
    all(c -> names(c) == names, args) ||
        throw(ArgumentError("chain names differ"))

    chains = chains(c1)
    all(c -> chains(c) == chains, args) ||
        throw(ArgumentError("sets of chains differ"))

    @assert c1.logevidence == c2.logevidence
    @assert range(c1) == range(c2)

    chn = Chains(
        cat(c1.value, c2.value, dims=1),
        names(c1),
        c1.name_map,
        start=range(c1).start,
        thin=range(c1).step,
        evidence = c1.logevidence
    )
    # chn = Chain(c1.weight,
    #             cat(c1.value, c2.value, dims=1),
    #             c1.range,
    #             c1.names,
    #             c1.chains,
    #             merge(c1.info, c2.info)
    #     )
    return chn
end

function save!(c::Chains, spl::Sampler, model, vi)
    c.info[:spl] = spl
    c.info[:model] = model
    c.info[:vi] = deepcopy(vi)
    return c
end

function resume(c::Chains, n_iter::Int)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"
    return sample(  c.info[:model],
                    c.info[:spl].alg;    # this is actually not used
                    resume_from=c,
                    reuse_spl_n=n_iter
                  )
end
