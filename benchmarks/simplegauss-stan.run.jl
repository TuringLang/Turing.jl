using Mamba, Stan

include("simplegauss-stan.data.jl")
include("simplegauss-stan.model.jl")

simplegaussstan = Stanmodel(name="simplegauss", model=simplegaussstanmodel, nchains=1);

simple_gauss_stan_sim = stan(simplegaussstan, simplegaussstandata, CmdStanDir=CMDSTAN_HOME)
describe(simple_gauss_stan_sim)

s_stan = simple_gauss_stan_sim[1:1000, ["s"], :].value[:]
m_stan = simple_gauss_stan_sim[1:1000, ["m"], :].value[:]

println("Correctness check for Stan:")

print("  1. s ≈ 49/24 (ϵ = 0.15)")
ans1 = abs(mean(s_stan) - 49/24) <= 0.15
if ans1
  print_with_color(:green, " ✓\n")
else
  print_with_color(:red, " X\n")
  print_with_color(:red, "    s = $(mean(simple_gauss_sim[:s])), diff = $(abs(mean(simple_gauss_sim[:s]) - 49/24))\n")
end

print("  2. m ≈ 7/6 (ϵ = 0.15)")
ans2 = abs(mean(m_stan) - 7/6) <= 0.15
if ans2
  print_with_color(:green, "   ✓\n")
else
  print_with_color(:red, "   X\n")
  print_with_color(:red, "     m = $(mean(simple_gauss_sim[:m])), diff = $(abs(mean(simple_gauss_sim[:m]) - 7/6))\n")
end



s_turing = simple_gauss_sim[:s]
m_turing = simple_gauss_sim[:m]


println("The difference between the means from Stan and Turing is:")
println("  E(s_stan) - E(s_turing) = $(mean(s_stan) - mean(s_turing))")
println("  E(m_stan) - E(m_turing) = $(mean(m_stan) - mean(m_turing))")
