using Turing, Test

N = 5
alg = HMC(0.2, 4)

@model function vdemo((::Type{T})=Float64) where {T}
    v = Vector{Matrix{T}}(undef, N)
    @. v ~ Wishart(7, [1 0.5; 0.5 1])
end

t_vec = @elapsed res_vec = sample(vdemo(), alg, 1000)

@model function vdemo()
    v = Vector{Matrix{Real}}(undef, N)
    for i in 1:N
        v[i] ~ Wishart(7, [1 0.5; 0.5 1])
    end
end

t_loop = @elapsed res = sample(vdemo(), alg, 1000)
