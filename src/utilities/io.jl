#########################
# Sampler I/O Interface #
#########################

##########
# Sample #
##########

mutable struct SampleInfo
    lf_num::Int
    elapsed::Float64
    epsilon::Float64
    eval_num::Int
    lf_eps::Float64
    le::Float64
end
SampleInfo() = SampleInfo(0, NaN, NaN, 0, NaN, NaN)

mutable struct Sample{Tvalue}
    weight :: Float64     # particle weight
    info   :: SampleInfo
    value  :: Tvalue
end
Sample(weight, value::Union{NamedTuple, Dict}) = Sample(weight, SampleInfo(), value)

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

"""
A wrapper of output trajactory of samplers.

Example:

```julia
# Define a model
@model xxx begin
  ...
  return(mu,sigma)
end

# Run the inference engine
chain = sample(xxx, SMC(1000))

chain[:logevidence]   # show the log model evidence
chain[:mu]            # show the weighted trajactory for :mu
chain[:sigma]         # show the weighted trajactory for :sigma
mean(chain[:mu])      # find the mean of :mu
mean(chain[:sigma])   # find the mean of :sigma
```
"""
struct Chain{R<:AbstractRange{Int}} <: AbstractChains
    weight  ::  Float64                 # log model evidence
    value   ::  Array{Union{Missing, Float64}, 3}
    range   ::  R # TODO: Perhaps change to UnitRange?
    names   ::  Vector{String}
    chains  ::  Vector{Int}
    info    ::  Dict{Symbol,Any}
end

function Chain()
    return Chain{StepRange{Int, Int}}( 0.0,
                                      Array{Float64, 3}(undef, 0, 0, 0),
                                      0:0,
                                      Vector{String}(),
                                      Vector{Int}(),
                                      Dict{Symbol,Any}()
                                    )
end

function Chain(w::Real, s::AbstractArray{<:Sample})
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
    return chn
end

# ind2sub is deprecated in Julia 1.0
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

function flatten(s::Sample{<:Dict})
    vals  = Vector{Float64}()
    names = Vector{String}()
    for (k, v) in s.value
        flatten(names, vals, string(k), v)
    end
    return Dict(names[i] => vals[i] for i in 1:length(vals))
end
function flatten(s::Sample{<:NamedTuple})
    vals  = Vector{Float64}()
    names = Vector{String}()
    for f in fieldnames(typeof(s.value))
        field = getfield(s.value, f)
        if field isa Dict
            for (k, v) in field
                flatten(names, vals, string(k), v)
            end
        else
            flatten(names, vals, string(f), field)
        end
    end
    for f in fieldnames(typeof(s.info))
        field = getfield(s.info, f)
        flatten(names, vals, string(f), field)
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

function save(c::Chains, spl::Sampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end
