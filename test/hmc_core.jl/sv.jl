using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Turing: _hmc_step
using Turing
using HDF5, JLD
sv_data = load(TPATH*"/example-models/nips-2017/sv-data.jld.data")["data"]

@model sv_model(T, y) = begin
    ϕ ~ Uniform(-1, 1)
    σ ~ Truncated(Cauchy(0,5), 0, +Inf)
    μ ~ Cauchy(0, 10)
    # h = tzeros(Real, T)
    h = Vector{Real}(T)
    h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
    y[1] ~ Normal(0, exp.(h[1] / 2))
    for t = 2:T
      h[t] ~ Normal(μ + ϕ * (h[t-1] - μ) , σ)
      y[t] ~ Normal(0, exp.(h[t] / 2))
    end
  end


mf = sv_model(data=sv_data[1])
chain_nuts = sample(mf, HMC(2000, 0.05, 10))

println("mean of m: $(mean(chn[:μ][1000:end]))")










# θ_dim = 1
# function lj_func(θ)
#   _lj = zero(Real)

#   s = 1

#   m = θ[1]
#   _lj += logpdf(Normal(0, sqrt(s)), m)

#   _lj += logpdf(Normal(m, sqrt(s)), 2.0)
#   _lj += logpdf(Normal(m, sqrt(s)), 2.5)

#   return _lj
# end

# neg_lj_func(θ) = -lj_func(θ)
# const f_tape = GradientTape(neg_lj_func, randn(θ_dim))
# const compiled_f_tape = compile(f_tape)

# function grad_func(θ)

#   inputs = θ
#   results = similar(θ)
#   all_results = DiffResults.GradientResult(results)

#   gradient!(all_results, compiled_f_tape, inputs)

#   neg_lj = all_results.value
#   grad, = all_results.derivs

#   return -neg_lj, grad

# end

# std = ones(θ_dim)
# θ = randn(θ_dim)
# lj = lj_func(θ)

# chn = []
# accept_num = 1


# totla_num = 5000
# for iter = 1:totla_num
#   push!(chn, θ)
#   θ, lj, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, 5, 0.05, std)
#   accept_num += is_accept
#   # if (iter % 50 == 0) println(θ) end
# end

# @show mean(chn), lj
# @show accept_num / totla_num
