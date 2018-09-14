# More stable, faster version of rand(Categorical)
function randcat(p::Vector{Float64})
  # if(any(p .< 0)) error("Negative probabilities not allowed"); end
  r, s = rand(), one(Int)
  for j = 1:length(p)
    r -= p[j]
    if(r <= 0.0) s = j; break; end
  end

  s
end

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
