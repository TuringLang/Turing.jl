using Distributions, Turing

@model gdemo(x) = begin
  # s ~ InverseGamma(2,3)
  s = 1
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

alg = eNUTS(200, 0.01)
res = sample(gdemo([1.5, 2.0]), alg)
# print(mean(res[:m]))

ans1 = abs(mean(res[:m]) - 7/6) <= 0.15
print("E[m] ≈ $(7/6) ? ")
if ans1
  print_with_color(:green, " ✓\n")
else
  print_with_color(:red, " X\n")
  print_with_color(:red, "    m = $(mean(res[:m])), diff = $(abs(mean(res[:m]) - 7/6))\n")
end
