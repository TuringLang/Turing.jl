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
