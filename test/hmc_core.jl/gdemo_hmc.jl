using Turing: _hmc_step

include("unit_test_helper.jl")
include("gdemo.jl")

# Turing

mf = gdemo()
chn = sample(mf, HMC(2000, 0.05, 5))

println("mean of s: $(mean(chn[:s][1000:end]))")
println("mean of m: $(mean(chn[:m][1000:end]))")

# Plain Julia

std = ones(θ_dim)
θ = randn(θ_dim)
lj = lj_func(θ)

chn = Dict(:θ=>Vector{Vector{Float64}}(), :logϵ=>Vector{Float64}())
accept_num = 0

function dummy_print(args...)
  nothing
end

totla_num = 5000
for iter = 1:totla_num

  push!(chn[:θ], θ)
  θ, lj, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, 5, 0.05, std)
  accept_num += is_accept

end

@show lj
samples_s = exp.(map(x -> x[1], chn[:θ]))
samples_m = map(x -> x[2], chn[:θ])
@show mean(samples_s[1000:end])
@show mean(samples_m[1000:end])
@show std(samples_s[1000:end])
@show std(samples_m[1000:end])

@show accept_num / totla_num

# Unit tests
