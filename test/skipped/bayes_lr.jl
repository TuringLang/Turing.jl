using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Turing: _hmc_step
using Turing

@model bayes_lr(xs, ys) = begin
    N = length(xs)
    @assert N == length(ys)

    s = 1
    β ~ Normal(0, 1)

    for n = 1:N
        ys[n] ~ Normal(xs[n] * β, sqrt(s))
    end
end

N = 100
xs = collect(range(-10, stop = 10, length = N))
s = 1
β = rand(Normal(0, 1))
ys = xs * β + rand(Normal(0, sqrt(s)), N)

println("s=$s, β=$β")

mf = bayes_lr(xs, ys)
chn = sample(mf, HMC(0.005, 3), 2000)

println("mean of β: ", mean(chn[1000:end, :β]))

θ_dim = 1
function lj_func(θ)
  N = length(xs)
  _lj = zero(Real)

  s = 1

  β = θ[1]
  _lj += logpdf(Normal(0, 1), β)
  for n = 1:N
    _lj += logpdf(Normal(xs[n] * β, sqrt(s)), ys[n])
  end

  return _lj
end

neg_lj_func(θ) = -lj_func(θ)
const f_tape = GradientTape(neg_lj_func, randn(θ_dim))
const compiled_f_tape = compile(f_tape)

function grad_func(θ)

  inputs = θ
  results = similar(θ)
  all_results = DiffResults.GradientResult(results)

  ReverseDiff.gradient!(all_results, compiled_f_tape, inputs)

  neg_lj = all_results.value
  grad, = all_results.derivs

  return -neg_lj, grad

end

std = ones(θ_dim)
θ = randn(θ_dim)
lj = lj_func(θ)

chn = []
accept_num = 1

total_num = 2000
for iter = 1:total_num
  global θ, chn, lj, lj_func, grad_func, std, accept_num
  push!(chn, θ)
  θ, lj, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, 3, 0.005, std)
  accept_num += is_accept
#   if (iter % 50 == 0) println(θ) end
end

@show mean(chn[1000:end]), lj
@show accept_num / total_num
