using Turing: _hmc_step

include("unit_test_helper.jl")
include("simple_gauss.jl")

# Turing

mf = simple_gauss()
chn = sample(mf, HMC(10000, 0.05, 10))

println("mean of m: $(mean(chn[1000:end, :m]))")

# Plain Julia

std = ones(θ_dim)
θ = randn(θ_dim)
lj = lj_func(θ)

chn = Dict(:θ=>Vector{Vector{Float64}}(), :logϵ=>Vector{Float64}())
accept_num = 1


totla_num = 10000
for iter = 1:totla_num

  push!(chn[:θ], θ)
  θ, lj, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, 10, 0.05, std)
  accept_num += is_accept

end

@show lj
@show mean(chn[:θ])
samples_first_dim = map(x -> x[1], chn[:θ])
@show std(samples_first_dim)

@show accept_num / totla_num

# Unit tests
