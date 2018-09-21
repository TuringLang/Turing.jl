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


using Pkg;
"""
    isinstalled(x::String)
Check if a package is installed.
"""
isinstalled(x::AbstractString) = x ∈ keys(Pkg.installed())


# NOTE: Remove the code below when DynamicHMC is registered.
using Pkg;
isinstalled("DynamicHMC") || pkg"add https://github.com/tpapp/DynamicHMC.jl#master";
isinstalled("TransformVariables") || pkg"add https://github.com/tpapp/TransformVariables.jl";
isinstalled("LogDensityProblems") || pkg"add https://github.com/tpapp/LogDensityProblems.jl";
