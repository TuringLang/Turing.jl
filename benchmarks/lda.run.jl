using Distributions
using Turing
using Stan

using Requests
import Requests: get, post, put, delete, options, FileParam

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda.model.jl")

include(Pkg.dir("Turing")*"/benchmarks/"*"lda-stan.run.jl")

setchunksize(60)

#for alg in ["HMC(2000, 0.25, 10)", "HMCDA(1000, 0.65, 1.5)", "NUTS(2000, 1000, 0.65)"]
tbenchmark("HMC(20, 0.025, 10)", "ldamodel_vec", "data=ldastandata[1]") # first run for compilation

for (modelc, modeln) in zip(["ldamodel_vec", "ldamodel"], ["LDA-vec", "LDA"])
  bench_res = tbenchmark("HMC(2000, 0.025, 10)", modelc, "data=ldastandata[1]")
  bench_res[4].names = ["phi[1]", "phi[2]"]
  logd = build_logd(modeln, bench_res...)
  logd["stan"] = lda_stan_d
  logd["time_stan"] = lda_time
  print_log(logd)
  send_log(logd)
end
