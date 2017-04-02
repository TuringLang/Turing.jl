using Turing
using Distributions
using Base.Test

include("simplegauss.data.jl")
include("simplegauss.model.jl")
alg = Gibbs(2000, PG(30, 3, :s), HMC(2, 0.1, 3, :m))
sim = @sample(simplegaussmodel(simplegaussdata), alg)
describe(sim)

print("  1. s ≈ 49/24 (ϵ = 0.15)")
ans1 = abs(mean(sim[:s]) - 49/24) <= 0.15
if ans1
  print_with_color(:green, " ✓\n")
else
  print_with_color(:red, " X\n")
  print_with_color(:red, "    s = $(mean(sim[:s])), diff = $(abs(mean(sim[:s]) - 49/24))\n")
end

print("  2. m ≈ 7/6 (ϵ = 0.15)")
ans2 = abs(mean(sim[:m]) - 7/6) <= 0.15
if ans2
  print_with_color(:green, "   ✓\n")
else
  print_with_color(:red, "   X\n")
  print_with_color(:red, "     m = $(mean(sim[:m])), diff = $(abs(mean(sim[:m]) - 7/6))\n")
end
