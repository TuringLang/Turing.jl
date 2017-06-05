using Turing

@model ldamodel(K, V, M, N, w, doc, beta, alpha) = begin
  theta = Matrix{Real}(K, M)
  for m = 1:M
    theta ~ [Dirichlet(alpha)]
  end

  phi = Matrix{Real}(V, K)
  for k = 1:K
    phi ~ [Dirichlet(beta)]
  end
  
  phi_dot_theta = phi * theta
  for n = 1:N
    Turing.acclogp!(vi, phi_dot_theta[w[n], doc[n]])
  end
end

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")

setchunksize(60)

alg = "HMCDA(2000, 0.65, 1.5)"
bench_res = tbenchmark(alg, "ldamodel", "data=ldastandata[1]")
bench_res[4].names = ["phi[1]", "phi[2]"]
logd = build_logd("LDA", bench_res...)
# logd["stan"] = lda_stan_d
# logd["time_stan"] = lda_time
print_log(logd)
