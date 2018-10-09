# Resampling schemes for particle filters
# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# More stable, faster version of rand(Categorical)
function randcat(p::Vector{Float64})
    r, s = rand(), one(Int)
    for j in eachindex(p)
        r -= p[j]
        if r <= 0
            s = j
            break
        end
    end
    return s
end

function resampleSystematic(w::AbstractVector{<:Real}, N::Int)

    Q = cumsum(w)
    T = vcat(range(0, stop=maximum(Q) - 1 / N, length=N)) .+ rand() / N, 1)

    i, j, indx = 1, 1, Array{Int}(undef, N)
    while i <= N
        if T[i] < Q[j]
            indx[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indx
end
