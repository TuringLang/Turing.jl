using Distributions
using Turing
using Stan

using Requests
import Requests: get, post, put, delete, options, FileParam

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda.model.jl")

include(Pkg.dir("Turing")*"/benchmarks/"*"lda-stan.run.jl")
println("Stan time: ", lda_time)
# setchunksize(60)
setadbackend(:reverse_diff)
setadsafe(false)

# tbenchmark("HMC(2, 0.025, 10)", "ldamodel", "data=ldastandata[1]")

# turnprogress(false)

for (modelc, modeln) in zip([
  # "ldamodel_vec", 
  "ldamodel"
  ], [
    # "LDA-vec", 
    "LDA"
    ])
  tbenchmark("HMC(2, 0.025, 10)", modelc, "data=ldastandata[1]")
  bench_res = tbenchmark("HMC(3000, 0.025, 10)", modelc, "data=ldastandata[1]")
  bench_res[4].names = ["phi[$k]" for k in 1:ldastandata[1]["K"]]
  logd = build_logd(modeln, bench_res...)
  logd["stan"] = lda_stan_d
  logd["time_stan"] = lda_time
  print_log(logd)
  send_log(logd)
end
