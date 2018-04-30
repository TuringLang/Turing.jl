using Distributions, DiffBase
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Turing: _hmc_step

# @model simple_gauss() = begin
#     s = 1
#     m ~ Normal(0,sqrt(s))
#     2.0 ~ Normal(m, sqrt(s))
#     2.5 ~ Normal(m, sqrt(s))
# end

θ_dim = 1
function lj_func(θ)
  _lj = zero(Real)
  
  s = 1

  m = θ[1]
  _lj += logpdf(Normal(0, sqrt(s)), m)

  _lj += logpdf(Normal(m, sqrt(s)), 2.0)
  _lj += logpdf(Normal(m, sqrt(s)), 2.5)

  return _lj
end

neg_lj_func(θ) = -lj_func(θ)
const f_tape = GradientTape(neg_lj_func, randn(θ_dim))
const compiled_f_tape = compile(f_tape)

function grad_func(θ)
    
  inputs = θ
  results = similar(θ)
  all_results = DiffBase.GradientResult(results)

  gradient!(all_results, compiled_f_tape, inputs)

  neg_lj = all_results.value
  grad, = all_results.derivs

  return -neg_lj, grad

end

stds = ones(θ_dim)
θ = randn(θ_dim)
lj = lj_func(θ)

chn = []
accept_num = 1

function dummy_print(args...)
  nothing
end

totla_num = 5000
for iter = 1:totla_num
  push!(chn, θ)
  θ, lj, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, 5, 0.05, stds; dprint=dummy_print)
  accept_num += is_accept
  if (iter % 50 == 0) println(θ) end
end

@show mean(chn), lj
@show accept_num / totla_num