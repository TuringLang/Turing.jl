# Resampling schemes for particle filters
# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# default resampling scheme
function resample(w::Vector{Float64},
                  num_particles::Int = length(w))
    resampleSystematic(w, num_particles)
end

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

function resampleMultinomial(w::Vector{Float64},
                             num_particles::Int)

    s = Distributions.sampler(Categorical(w))
    indx = rand(s, num_particles)

end


function resampleResidual(w::Vector{Float64},
                          num_particles::Int)

    M = length( w )

    # "Repetition counts" (plus the random part, later on):
    Ns = floor.(length(w) .* w)

    # The "remainder" or "residual" count:
    R = Int(sum( Ns ))

    # The number of particles which will be drawn stocastically:
    M_rdn = num_particles-R;

    # The modified weights:
    Ws = (M .* w - floor.(M .* w))/M_rdn;

    # Draw the deterministic part:
    indx1 = Array{Int}(undef, R)
    i=1
    for j=1:M
        for k=1:Ns[j]
            indx1[i]=j
            i = i + 1
        end
    end

    # And now draw the stocastic (Multinomial) part:
    s = Distributions.sampler(Categorical(w))
    indx2 = rand(s, M_rdn)
    indx = append!(indx1, indx2)

    return indx

end

function resampleStratified(w::Vector{Float64},
                            num_particles::Int)

    N = num_particles
    Q = cumsum(w)

    T = Array{Float64}(undef, N+1)
    for i=1:N,
        T[i] = rand()/N + (i-1)/N
    end
    T[N+1] = 1

    i=1
    j=1

    indx = Array{Int}(undef, N)
    while i<=N
        if T[i]<Q[j]
            indx[i]=j
            i=i+1
        else
            j=j+1
        end
    end

    return indx

end

function resampleSystematic(w::Vector{Float64},
                            num_particles::Int)

    N = num_particles
    Q = cumsum(w)

    T = collect(range(0, stop = maximum(Q)-1/N, length = N)) .+ rand()/N
    push!(T, 1)

    i=1
    j=1

    indx = Array{Int}(undef, N)
    while i<=N
        if (T[i]<Q[j])
            indx[i]=j
            i=i+1
        else
            j=j+1
        end
        # if j == length(w)
        #   indx[i] = j
        #   break;
        # end
    end

    return indx

end
