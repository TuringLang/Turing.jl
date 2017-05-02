using Distributions
using Turing
using Base.Test

x = [1.5 2.0]

@model gibbstest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

check(chain) = begin
  print("  1. s ≈ 49/24 (ϵ = 0.2)")
  ans1 = abs(mean(chain[:s]) - 49/24) <= 0.2
  if ans1
    print_with_color(:green, " ✓\n")
    print_with_color(:green, "    s = $(mean(chain[:s])), diff = $(abs(mean(chain[:s]) - 49/24))\n")
  else
    print_with_color(:red, " X\n")
    print_with_color(:red, "    s = $(mean(chain[:s])), diff = $(abs(mean(chain[:s]) - 49/24))\n")
  end

  print("  2. m ≈ 7/6 (ϵ = 0.2)")
  ans2 = abs(mean(chain[:m]) - 7/6) <= 0.2
  if ans2
    print_with_color(:green, "   ✓\n")
    print_with_color(:green, "    m = $(mean(chain[:m])), diff = $(abs(mean(chain[:m]) - 7/6))\n")
  else
    print_with_color(:red, "   X\n")
    print_with_color(:red, "     m = $(mean(chain[:m])), diff = $(abs(mean(chain[:m]) - 7/6))\n")
  end
end

check(sample(gibbstest(x), Gibbs(1500, PG(30, 3, :s), HMC(1, 0.2, 4, :m))))
check(sample(gibbstest(x), PG(30, 2500)))
# check(sample(gibbstest(x), Gibbs(1000, HMC(3, 0.2, 1, :s), eNUTS(1, 0.2, :m))))
