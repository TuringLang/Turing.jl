using Turing, HDF5, JLD, Gadfly, DataFrames
using Mamba: summarystats

TPATH = Pkg.dir("Turing")

hmm_nuts_chain = load(TPATH*"/nips-2017/hmm/hmm-collapsed-NUTS(1000,200,0.65)-chain.jld")["chain"]
hmm_gibbs_chain = load(TPATH*"/nips-2017/hmm/hmm-uncollapsed-Gibbs(1000,PG(50,1,:y),NUTS(1,200,0.65,:phi,:theta))-chain.jld")["chain"]

sv_nuts_chain = load(TPATH*"/nips-2017/sv/new-first/chain-nuts.jld")["chain"]
sv_gibbs_chain = load(TPATH*"/nips-2017/sv/new-first/chain-gibbs.jld")["chain"]


lda_nuts_chain = load(TPATH*"/nips-2017/lda-exps-chain-3.jld")["chain"]
lda_gibbs_chain = load(TPATH*"/nips-2017/lda-exps-chain-3-un.jld")["chain"]

hmm_nuts_lps = hmm_nuts_chain[:lp]
hmm_gibbs_lps = hmm_gibbs_chain[:lp]
lda_nuts_lps = lda_nuts_chain[:lp]
lda_gibbs_lps = lda_gibbs_chain[:lp]
sv_nuts_lps = sv_nuts_chain[:lp]
sv_gibbs_lps = sv_gibbs_chain[:lp]

colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d", colorant"#c0392b"]

N = 1000
# hmm_lps_df = DataFrame(Samples=[collect(1:N);collect(1:N)], Engine=[["NUTS" for _ = 1:N]..., ["Gibbs" for _ = 1:N]...], Logp=[hmm_nuts_lps; hmm_gibbs_lps])
# hmm_lps_layer = layer(hmm_lps_df, x=:Samples, y=:Logp, color=:Engine, Geom.line)
# plot(hmm_lps_layer)

# hmm_nuts_lps_layer = layer(x=1:N, y=hmm_nuts_lps, Geom.line, Theme(default_color=colors[1]))
# hmm_gibbs_lps_layer = layer(x=1:N, y=hmm_gibbs_lps, Geom.line, Theme(default_color=colors[2]))

# lda_nuts_lps_layer
# lda_gibbs_lps_layer
# sv_nuts_lps_layer
# sv_gibbs_lps_layer

# lda_lps_df = DataFrame(Samples=[collect(1:N);collect(1:N)], Engine=[["NUTS" for _ = 1:N]..., ["Gibbs" for _ = 1:N]...], Logp=[lda_nuts_lps; lda_gibbs_lps])
# lda_lps_layer = layer(lda_lps_df, x=:Samples, y=:Logp, color=:Engine, Geom.line)
#
# sv_lps_df = DataFrame(Samples=[collect(20:N);collect(20:N)], Engine=[["NUTS" for _ = 20:N]..., ["Gibbs" for _ = 20:N]...], Logp=[sv_nuts_lps[20:N]; sv_gibbs_lps[20:N]])
# sv_lps_layer = layer(sv_lps_df, x=:Samples, y=:Logp, color=:Engine, Geom.line)
# plot(sv_lps_layer)

SV_start = 10
lps_df = DataFrame(
  Samples=[collect(1:N); collect(1:N);
           collect(1:N); collect(1:N);
           collect(SV_start:N); collect(SV_start:N)],
  Engine=[["NUTS" for _ = 1:N]..., ["Gibbs" for _ = 1:N]...,
          ["NUTS" for _ = 1:N]..., ["Gibbs" for _ = 1:N]...,
          ["NUTS" for _ = SV_start:N]..., ["Gibbs" for _ = SV_start:N]...],
  Logp=[hmm_nuts_lps; hmm_gibbs_lps;
        lda_nuts_lps; lda_gibbs_lps;
        sv_nuts_lps[SV_start:N]; sv_gibbs_lps[SV_start:N]],
  Model=[["HMM" for _ = 1:N]..., ["HMM" for _ = 1:N]...,
         ["LDA" for _ = 1:N]..., ["LDA" for _ = 1:N]...,
         ["SV" for _ = SV_start:N]..., ["SV" for _ = SV_start:N]...]
)

lps_plot_all = plot(lps_df, xgroup="Model", x="Samples", y="Logp", color="Engine", Geom.subplot_grid(Coord.cartesian(ymin=-1200, ymax=0), Geom.line, free_y_axis=true),
                    # Guide.title(join(["r=1","r=2"], " "^48)),
                    Guide.xlabel("Number of iterations"), Guide.ylabel("Log-joint"))

Gadfly.push_theme(:default)
draw(PDF(TPATH*"/nips-2017/pls-all.pdf", 7inch, 3inch), lps_plot_all)

describe(sv_gibbs_chain)
