using Distributions, Turing

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  # s = 1
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

alg = eNUTS(5000, 0.1)
res = sample(gdemo([1.5, 2.0]), alg)
# print(mean(res[:m]))

eps = 0.05
ans1 = abs(mean(res[:m]) - 7/6) <= eps
print("E[m] = $(mean(res[:m])) ≈ $(7/6) (ϵ = $eps) ? ")
if ans1
  print_with_color(:green, " ✓\n")
else
  print_with_color(:red, " X\n")
  print_with_color(:red, "    m = $(mean(res[:m])), diff = $(abs(mean(res[:m]) - 7/6))\n")
end

print("E[s] = $(mean(res[:s])) ≈ $(49/24) (ϵ = $eps) ?")
ans2 = abs(mean(res[:s]) - 49/24) <= eps
if ans2
  print_with_color(:green, "  ✓\n")
else
  print_with_color(:red, "   X\n")
  print_with_color(:red, "     s = $(mean(res[:s])), diff = $(abs(mean(res[:s]) - 49/24))\n")
end
