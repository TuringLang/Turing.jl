using Turing, HDF5, JLD, DataFrames
using Mamba: summarystats

TPATH = Pkg.dir("Turing")

data = 4

smr_all = Dict()

g_ess_all = []
g_time_all = []

for run  = [1,2]
    g_chain = load(TPATH*"/nips-2017/sv/chain-gibbs-data-$data-run-$run.jld")["chain"]

    g_smr = summarystats(g_chain)
    smr_all["g-$run"] = g_smr

    ess_idx = findfirst(g_smr.colnames, "ESS")
    lp_idx = findfirst(g_smr.rownames, "lp")

    g_ess = g_smr.value[lp_idx,ess_idx,1]; push!(g_ess_all, g_ess)

    g_time = sum(g_chain[:elapsed]); push!(g_time_all, g_time)
end

n_ess_all = []
n_time_all = []

for run  = [1,2]
    n_chain = load(TPATH*"/nips-2017/sv/chain-nuts-data-$data-run-$run.jld")["chain"]

    n_smr = summarystats(n_chain)
    smr_all["n-$run"] = n_smr

    ess_idx = findfirst(n_smr.colnames, "ESS")
    lp_idx = findfirst(n_smr.rownames, "lp")

    n_ess = n_smr.value[lp_idx,ess_idx,1]; push!(n_ess_all, n_ess)

    n_time = sum(n_chain[:elapsed]); push!(n_time_all, n_time)
end

df = DataFrame(Engine = [["Gibbs" for _ = 1:length(g_ess_all)]..., ["NUTS" for _ = 1:length(n_ess_all)]...],
               ESS    = [g_ess_all; n_ess_all],
               Time   = [g_time_all; n_time_all])

save(TPATH*"/nips-2017/sv/sv-data-$data-df.jld", "df", df)
save(TPATH*"/nips-2017/sv/sv-data-$data-smr.jld", "smr_all", smr_all)
