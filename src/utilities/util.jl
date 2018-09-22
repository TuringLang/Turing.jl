"""
    data(dict::Dict, keys::Vector{Symbol})
Construct a tuple with values filled according to `dict` and keys
according to `keys`.
"""
function data(dict::Dict, keys::Vector{Symbol})

    @assert mapreduce(k -> haskey(dict, k), &, keys)

    r = Expr(:tuple)
    for k in keys
        push!(r.args, Expr(:(=), k, dict[k]))
    end
    return Main.eval(r)
end
