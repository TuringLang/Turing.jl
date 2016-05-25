import ForwardDiff                    # for graident

function hmcpdf(pdf::Function, x, params...)
  return pdf(params, x)
end

function hmcMvNormal(μ, Σ, x)
  Λ = inv(Σ)
  return 1 / sqrt((2pi) ^ 2 * det(Σ)) * exp(-0.5 * ((x - μ)' * Λ * (x - μ))[1])
end

function toyDensity(x, y)
  return x * y
end








function fun(x, y, z)
  return x^2 + 2y + sqrt(z)
end

∇fun = ForwardDiff.gradient(fun)
fun(1, 1, 1)

function fun2(x, y)
  return hmcpdf(toyDensity, x, y)
end

∇fun2 = ForwardDiff.gradient(fun2)
fun2(1.0, 2.0)
