using Distributions
using Turing
using Mamba: summarystats

TPATH = Pkg.dir("Turing")

include(TPATH*"/benchmarks/benchmarkhelper.jl")
include(TPATH*"/example-models/stan-models/lda-stan.data.jl")

collapsed = true

@model ldamodel(K, V, M, N, w, doc, beta, alpha) = begin
  theta = Vector{Vector{Real}}(M)
  for m = 1:M
    theta[m] ~ Dirichlet(alpha)
  end

  phi = Vector{Vector{Real}}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(beta)
  end

  if collapsed
    phi_dot_theta = [log([dot(map(p -> p[i], phi), theta[m]) for i = 1:V]) for m=1:M]
    for n = 1:N
      Turing.acclogp!(vi, phi_dot_theta[doc[n]][w[n]])
    end
  else
    z = tzeros(Int, N)
    for n = 1:N
      z[n] ~ Categorical(theta[doc[n]])
      w[n] ~ Categorical(phi[z[n]])
    end
  end
end

include(TPATH*"/benchmarks/"*"lda-stan.run.jl")

setchunksize(60)

include(TPATH*"/nips-2017/"*"lda-settings.jl")

res = Dict()
res[true] = Dict()
res[false] = Dict()

for iscollapsed = [true,false]

  collapsed = iscollapsed

  res[collapsed][:t_elapsed] = Vector{Float64}(N)
  res[collapsed][:Min_ESS] = Vector{Float64}(N)
  res[collapsed][:Max_MCSE] = Vector{Float64}(N)

  for i = 1:N
    alg = iscollapsed ? spls[i] : spls_un[i]

    bench_res = tbenchmark(alg, "ldamodel", "data=ldastandata[1]")
    smr = summarystats(bench_res[4])

    ess_idx = findfirst(smr.colnames, "ESS")
    min_ess = min(smr.value[:,ess_idx,1]...)

    mcse_idx = findfirst(smr.colnames, "MCSE")
    min_mcse = min(smr.value[:,mcse_idx,1]...)

    chain = bench_res[4]
    # describe(chain)
    bench_res[4].names = ["phi[1]", "phi[2]"]

    logd = build_logd("LDA", bench_res...)
    logd["stan"] = lda_stan_d
    logd["time_stan"] = lda_time
    print_log(logd)

    # Record time, mini ess and max mcse
    res[collapsed][:t_elapsed][i] = logd["time"]
    res[collapsed][:Min_ESS][i] = min_ess
    res[collapsed][:Max_MCSE][i] = min_mcse

    # Save logp
    lps = chain[:lp]
    if iscollapsed
      writedlm(TPATH*"/nips-2017/lda-exps-lp-$i.txt", lps)
    else
      writedlm(TPATH*"/nips-2017/lda-exps-lp-$i-un.txt", lps)
    end

  end

end

using DataFrames

# Save Stan statistics
df = DataFrame(Sampler = ["NUTS(1000,0.65)"], Time_elpased = lda_time)
writetable(TPATH*"/nips-2017/lda-exps-summary-stan.csv", df)

# Save Turing statistics
df = DataFrame(Sampler = spls, Time_elpased = res[true][:t_elapsed], Min_ESS = res[true][:Max_MCSE], Max_MCSE = res[true][:Max_MCSE])
writetable(TPATH*"/nips-2017/lda-exps-summary.csv", df)

df = DataFrame(Sampler = spls_un, Time_elpased = res[false][:t_elapsed], Min_ESS = res[false][:Max_MCSE], Max_MCSE = res[false][:Max_MCSE])
writetable(TPATH*"/nips-2017/lda-exps-summary-un.csv", df)

############
# Plotting #
############

include(TPATH*"/nips-2017/lda-plots.jl")
