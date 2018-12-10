# Resampling schemes for particle filters
# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# default resampling scheme
function resample(w::AbstractVector{<:Real}, num_particles::Integer=length(w))
    return resample_systematic(w, num_particles)
end

# More stable, faster version of rand(Categorical)
function randcat(p::AbstractVector{T}) where T<:Real
    r, s = rand(T), 1
    for j in eachindex(p)
        r -= p[j]
        if r <= zero(T)
            s = j
            break
        end
    end
    return s
end

function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer)

    M = length(w)

    # "Repetition counts" (plus the random part, later on):
    Ns = floor.(length(w) .* w)

    # The "remainder" or "residual" count:
    R = Int(sum(Ns))

    # The number of particles which will be drawn stocastically:
    M_rdn = num_particles - R

    # The modified weights:
    Ws = (M .* w - floor.(M .* w)) / M_rdn

    # Draw the deterministic part:
    indx1, i = Array{Int}(undef, R), 1
    for j in 1:M
        for k in 1:Ns[j]
            indx1[i] = j
            i += 1
        end
    end

    # And now draw the stocastic (Multinomial) part:
    return append!(indx1, rand(Distributions.sampler(Categorical(w)), M_rdn))
end

function resample_stratified(w::AbstractVector{<:Real}, num_particles::Integer)

    Q, N = cumsum(w), num_particles

    T = Array{Float64}(undef, N + 1)
    for i=1:N,
        T[i] = rand() / N + (i - 1) / N
    end
    T[N+1] = 1

    indx, i, j = Array{Int}(undef, N), 1, 1
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

function resample_systematic(w::AbstractVector{<:Real}, num_particles::Integer)
    Q, N = cumsum(w), num_particles
    T = if N != 1
        collect(range(0, stop = maximum(Q)-1/N, length = N))
    else
        zeros(N)
    end

    T .+= rand()/N
    push!(T, 1)

    indx, i, j = Array{Int}(undef, N), 1, 1
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
