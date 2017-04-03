using Distributions
using Turing
using Base.Test

x = [1.5 2.0]

@model gibbstest(x) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

gibbs = Gibbs(2000, PG(30, 3, :s), HMC(2, 0.1, 7, :m))
chain = sample(gibbstest(x), gibbs)

print("  1. s ≈ 49/24 (ϵ = 0.15)")
ans1 = abs(mean(chain[:s]) - 49/24) <= 0.15
if ans1
  print_with_color(:green, " ✓\n")
else
  print_with_color(:red, " X\n")
  print_with_color(:red, "    s = $(mean(chain[:s])), diff = $(abs(mean(chain[:s]) - 49/24))\n")
end

print("  2. m ≈ 7/6 (ϵ = 0.15)")
ans2 = abs(mean(chain[:m]) - 7/6) <= 0.15
if ans2
  print_with_color(:green, "   ✓\n")
else
  print_with_color(:red, "   X\n")
  print_with_color(:red, "     m = $(mean(chain[:m])), diff = $(abs(mean(chain[:m]) - 7/6))\n")
end
