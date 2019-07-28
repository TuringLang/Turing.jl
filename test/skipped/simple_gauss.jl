include("unit_test_helper.jl")

# Turing

using Turing

@model simple_gauss() = begin
    s = 1
    m ~ Normal(0, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    2.5 ~ Normal(m, sqrt(s))
end

# Plain Julia

using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

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
  all_results = DiffResults.GradientResult(results)

  gradient!(all_results, compiled_f_tape, inputs)

  neg_lj = all_results.value
  grad, = all_results.derivs

  return -neg_lj, grad

end

# Unit test for gradient

test_grad(simple_gauss, grad_func)
