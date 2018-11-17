using Main.Test
using Turing: _nuts_step

include("unit_test_helper.jl")
include("gdemo.jl")

include("dual_averaging.jl")

# Turing

# mf = gdemo()
# chn = sample(mf, HMC(5000, 0.05, 5))

# println("mean of m: $(mean(chn[:m][1000:end]))")

# Plain Julia

M_adapt = 1000
ϵ0 = 0.25
logϵ = log(ϵ0)
μ = log(10 * ϵ0)
logϵbar = log(1)
Hbar = 0

δ = 0.75

for test_id = 1:2

  test_name =  "$test_id. NUTS " * (test_id == 1 ? "without DA" : "with DA")

  @testset "$test_name" begin

    std = ones(θ_dim)
    θ = randn(θ_dim)
    lj = lj_func(θ)

    chn = Dict(:θ=>Vector{Vector{Float64}}(), :logϵ=>Vector{Float64}())

    function dummy_print(args...)
      nothing
    end

    println("Start to run NUTS")

    totla_num = 10000
    for iter = 1:totla_num
      
      θ, da_stat = _nuts_step(θ, exp(logϵ), lj_func, grad_func, std)
      if test_id == 1
        logϵ, Hbar, logϵbar = _adapt_ϵ(logϵ, Hbar, logϵbar, da_stat, iter, M_adapt, δ, μ)
      end

      push!(chn[:θ], θ)
      push!(chn[:logϵ], logϵ)
      # if (iter % 50 == 0) println(θ) end
    end

    samples_s = exp.(map(x -> x[1], chn[:θ]))
    samples_m = map(x -> x[2], chn[:θ])
    @show mean(samples_s[1000:end])
    @test mean(samples_s[1000:end]) ≈ 49/24 atol=0.2
    @show mean(samples_m[1000:end])
    @test mean(samples_m[1000:end]) ≈ 7/6 atol=0.2

    @show std(samples_s[1000:end])
    @show std(samples_m[1000:end])

    @show mean(exp.(chn[:logϵ]))

  end

end
