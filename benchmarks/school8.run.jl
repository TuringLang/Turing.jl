using Turing
using Mamba: describe

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/school8.model.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/school8-stan.data.jl")

data = deepcopy(schools8data[1])
delete!(data, "tau")

# chn = sample(school8(data=data), HMC(2000, 0.75, 5))

bench_res = tbenchmark("HMC(2000, 0.75, 5)", "school8", "data=data")
# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("School 8", bench_res...)

# describe(chn)

include(Pkg.dir("Turing")*"/benchmarks/school8-stan.run.jl")

logd["stan"] = stan_d
logd["time_stan"] = get_stan_time("schools8")

print_log(logd)

using Requests
import Requests: get, post, put, delete, options, FileParam
send_log(logd)
