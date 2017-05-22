using Turing, HDF5, JLD, DataFrames
using Mamba: summarystats

TPATH = Pkg.dir("Turing")

data = 2.5

smr_all = Dict()

g_ess_all = []
g_time_all = []

for run  = [1,2,3,4,5]
    println("run $run for gibbs")
    g_chain = load(TPATH*"/nips-2017/gmm/gibbs-chain-$data-$run.jld")["chain"]

    g_smr = summarystats(g_chain)
    smr_all["g-$run"] = g_smr

    ess_idx = findfirst(g_smr.colnames, "ESS")
    x_idx = findfirst(g_smr.rownames, "x")

    g_ess = g_smr.value[x_idx,ess_idx,1]; push!(g_ess_all, g_ess)

    g_time = sum(g_chain[:elapsed]); push!(g_time_all, g_time)
end

n_ess_all = []
n_time_all = []

for run  = [1,2,3,4,5]
    println("run $run for nuts")
    n_chain = load(TPATH*"/nips-2017/gmm/nuts-chain-$data-$run.jld")["chain"]

    n_smr = summarystats(n_chain)
    smr_all["n-$run"] = n_smr

    ess_idx = findfirst(n_smr.colnames, "ESS")
    x_idx = findfirst(n_smr.rownames, "x")

    n_ess = n_smr.value[x_idx,ess_idx,1]; push!(n_ess_all, n_ess)

    n_time = sum(n_chain[:elapsed]); push!(n_time_all, n_time)
end

df_all = DataFrame(Run = [collect(1:length(g_ess_all))..., collect(1:length(n_ess_all))...],
                   Engine = [["Gibbs" for _ = 1:length(g_ess_all)]..., ["NUTS" for _ = 1:length(n_ess_all)]...],
                   ESS    = [g_ess_all; n_ess_all],
                   Time   = [g_time_all; n_time_all])

save(TPATH*"/nips-2017/gmm/gmm-$data-df.jld", "df", df_all)
save(TPATH*"/nips-2017/gmm/gmm-$data-smr.jld", "smr", smr_all)
