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
const _internal_vars = [
    "elapsed", "eval_num", "lf_eps", "lp",
    "acceptance_rate", "hamiltonian_energy", "is_accept", "log_density", "n_steps", "numerical_error", "step_size", "tree_depth",
]

function Chain(w::Real, s::AbstractArray{Sample})
    samples = flatten.(s)
    names_ = collect(mapreduce(s -> keys(s), union, samples))

    values_ = mapreduce(v -> map(k -> haskey(v, k) ? v[k] : missing, names_), hcat, samples)
    values_ = convert(Array{Union{Missing, Float64}, 2}, values_')

    chn = Chains(
        reshape(values_, size(values_, 1), size(values_, 2), 1),
        names_,
        Dict(:internals => _internal_vars),
        evidence = w,
        info = (samples = s,)
    )
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

function flatten(names, value :: AbstractArray{Float64}, k :: String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, AbstractArray)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                name = k * name
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

function save(c::Chains, spl::AbstractSampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

function resume(c::Chains, n_iter::Int)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"
    return sample(
        c.info[:model],
        c.info[:spl].alg;    # this is actually not used
        resume_from=c,
        reuse_spl_n=n_iter
    )
end

function split_var_str(var_str)
    ind = findfirst(c -> c == '[', var_str)
    inds = Vector{String}[]
    if ind == nothing
        return var_str, inds
    end
    sym = var_str[1:ind-1]
    ind = length(sym)
    while ind < length(var_str)
        ind += 1
        @assert var_str[ind] == '['
        push!(inds, String[])
        while var_str[ind] != ']'
            ind += 1
            if var_str[ind] == '['
                ind2 = findnext(c -> c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2]))
                ind = ind2+1
            else
                ind2 = findnext(c -> c == ',' || c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2-1]))
                ind = ind2
            end
        end
    end
    return sym, inds
end
