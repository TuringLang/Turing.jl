using Distributions, Turing, HDF5, JLD, StatsFuns
using Gadfly, DataFrames, Colors

Gadfly.push_theme(:default)
n = 3
colors = distinguishable_colors(n, LCHab[LCHab(70, 60, 240)],
                       transform=c -> deuteranopic(c, 0.5),
                       lchoices=Float64[65, 70, 75, 80],
                       cchoices=Float64[0, 50, 60, 70],
                       hchoices=linspace(0, 330, 24))

TPATH = Pkg.dir("Turing")

dfs = [load("/home/kai/sv-data-$i-df.jld")["df"] for i = 1:4]

ess_mean_g = []
time_mean_g = []
ess_mean_n = []
time_mean_n = []
ess_mins_g = []
time_mins_g = []
ess_maxs_g = []
time_maxs_g = []
ess_mins_n = []
time_mins_n = []
ess_maxs_n = []
time_maxs_n = []

for i = 1:4
  df = dfs[i]

  p_ess = plot(df, xgroup="Run", x="Engine", y="ESS", Geom.subplot_grid(Geom.bar, Guide.title(nothing),Guide.xlabel(nothing), ),Guide.xlabel(nothing))
  p_time = plot(df, xgroup="Run", x="Engine", y="Time", Geom.subplot_grid(Geom.bar,Guide.xlabel(nothing)),Guide.xlabel(nothing),)

  ess_gibbs = df[df[:Engine].=="Gibbs", :ESS][1:5]
  ess_nuts = df[df[:Engine].=="NUTS", :ESS][1:5]
  ess_mean = [mean(ess_gibbs), mean(ess_nuts)]; push!(ess_mean_g, ess_mean[1]); push!(ess_mean_n, ess_mean[2])
  ess_std = [std(ess_gibbs), std(ess_nuts)]
  n = 5
  ess_mins = ess_mean .- (1.96 * ess_std / sqrt(n)); push!(ess_mins_g, ess_mins[1]); push!(ess_mins_n, ess_mins[2])
  ess_maxs = ess_mean .+ (1.96 * ess_std / sqrt(n)); push!(ess_maxs_g, ess_maxs[1]); push!(ess_maxs_n, ess_maxs[2])


  time_gibbs = df[df[:Engine].=="Gibbs", :Time][1:5]
  time_nuts = df[df[:Engine].=="NUTS", :Time][1:5]
  time_mean = [mean(time_gibbs), mean(time_nuts)]; push!(time_mean_g, time_mean[1]); push!(time_mean_n, time_mean[2])
  time_std = [std(time_gibbs), std(time_nuts)]
  n = 5
  time_mins = time_mean .- (1.96 * time_std / sqrt(n)); push!(time_mins_g, time_mins[1]); push!(time_mins_n, time_mins[2])
  time_maxs = time_mean .+ (1.96 * time_std / sqrt(n)); push!(time_maxs_g, time_maxs[1]); push!(time_maxs_n, time_maxs[2])
end

g_locs = [0.25, 0.75, 1.25, 1.75] - 0.05
gibbs_layer_ess = layer(x = g_locs, xmin = g_locs - 0.05, xmax = g_locs + 0.05,
                       y = ess_mean_g, ymax = ess_maxs_g, ymin = ess_mins_g,
                       Geom.bar, Geom.errorbar, Theme(default_color = colors[1]))

n_locs = [0.25, 0.75, 1.25, 1.75] + 0.05
nuts_layer_ess = layer(x = n_locs, xmin = n_locs - 0.05, xmax = n_locs + 0.05,
                       y = ess_mean_n, ymax = ess_maxs_n, ymin = ess_mins_n,
                       Geom.bar, Geom.errorbar, Theme(default_color = colors[2]))

ess_sv_p = plot(gibbs_layer_ess, nuts_layer_ess,
                Coord.cartesian(ymin = 0), Guide.ylabel(nothing), Guide.xlabel(nothing), Guide.xticks(label=false),
                Guide.manual_color_key("", ["NUTS","Gibbs"], colors[[2,1]]))

gibbs_layer_time = layer(x = g_locs, xmin = g_locs - 0.05, xmax = g_locs + 0.05,
                         y = time_mean_g, ymax = time_maxs_g, ymin = time_mins_g,
                         Geom.bar, Geom.errorbar, Theme(default_color = colors[1]))

nuts_layer_time = layer(x = n_locs, xmin = n_locs - 0.05, xmax = n_locs + 0.05,
                        y = time_mean_n, ymax = time_maxs_n, ymin = time_mins_n,
                        Geom.bar, Geom.errorbar, Theme(default_color = colors[2]))

time_sv_p = plot(gibbs_layer_time, nuts_layer_time,
                 Coord.cartesian(ymin = 0, xmin = 0, xmax = 2.13), Guide.ylabel(nothing), Guide.xlabel(nothing), Guide.xticks(label=false),
                #  Guide.manual_color_key("", ["NUTS","Gibbs"], colors[[2,1]])
                 )

vs = vstack(ess_sv_p, time_sv_p)

draw(PDF(TPATH*"/nips-2017/sv/sv-ess-time.pdf", 8inch, 3inch), vs)



# i=3;ess_mins
# layer()
# plot(x=[0.5, 1.5], y=ess_mean, ymin=ess_mins, ymax=ess_maxs, xmin=[0.25, 1.25], xmax=[0.75, 1.75],
#              Geom.bar, Geom.errorbar,Coord.cartesian(ymin=0, ymax=7500),
#              Guide.yticks(label=i==1), Guide.ylabel(i==1?"ESS":""), Guide.xlabel(""))
# gs = gridstack([p_ess_all[1] p_ess_all[2] p_ess_all[3] p_ess_all[4];
#                 p_time_all[1] p_time_all[2] p_time_all[3] p_time_all[4]])

#
# draw(PDF(TPATH*"/nips-2017/sv/sv-ess-time.pdf", 8inch, 4inch), gs)
#
#

#
# plot(dataset("datasets", "OrchardSprays"),
#      xgroup="Treatment", x="ColPos", y="RowPos", color="Decrease",
#      Geom.subplot_grid(Geom.point))

# sds = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
# n = 10
# ys = [mean(rand(Normal(0, sd), n)) for sd in sds]
# ymins = ys .- (1.96 * sds / sqrt(n))
# ymaxs = ys .+ (1.96 * sds / sqrt(n))
#
# plot(x=1:length(sds), y=ys, ymin=ymins, ymax=ymaxs,
#      Geom.point, Geom.errorbar)



















#
# spl_colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d", colorant"#c0392b"]
# lps_gibbs = chain_gibbs[:lp]
#
# l1 = layer(x=15:1000, y=-lps_gibbs[15:end], Geom.line, Geom.line,Theme(default_color=spl_colors[1]))
#
# lps_nuts = chain_nuts[:lp]
#
# l2 = layer(x=15:1000, y=-lps_nuts[15:end], Geom.line, Geom.line,Theme(default_color=spl_colors[2]))
#
# lp_plot = plot(l1, l2,
# Guide.xlabel("Number of iterations"), Guide.ylabel("Negative log-posterior"),
# Guide.title("Negative Log-posterior for the Stochastic Volatility Model"), Guide.manual_color_key("Legend", ["Gibbs", "NUTS"], spl_colors[1:2]))
#
# draw(PDF(TPATH*"/nips-2017/sv/lp_plot.pdf", 8inch, 4.5inch), lp_plot)
#
# describe(chain_gibbs)
# describe(chain_nuts)
#
#
#
# #
# #
# #
# #
# #
# #
# # NNNNEW
# #
# #
# #
# #
# #
# #
# #
#
# sv_nuts_1_1 = load("/home/kai/sv-nuts-1-1.jld")["chain"]
# sv_gibbs_1_1 = load("/home/kai/sv-gibbs-1-1.jld")["chain"]
#
# sv_gibbs_1_2 = load("/home/kai/sv-gibbs-1-2.jld")["chain"]
# sv_gibbs_1_3 = load("/home/kai/sv-gibbs-1-3.jld")["chain"]
#
#
# lp_nuts_1_1 = sv_nuts_1_1[:lp]
# lp_gibbs_1_1 = sv_gibbs_1_1[:lp]
# lp_gibbs_1_2 = sv_gibbs_1_2[:lp]
# lp_gibbs_1_3 = sv_gibbs_1_3[:lp]
#
# using DataFrames
# N = 10000
# df_trace = DataFrame(Samples=[collect(1:N); collect(1:N); collect(1:N); collect(1:N)],
#                      Engine=[["Gibbs 1" for _ = 1:N]..., ["Gibbs 2" for _ = 1:N]..., ["Gibbs 3" for _ = 1:N]..., ["NUTS" for _ = 1:N]...],
#                      lp=[lp_gibbs_1_1; lp_gibbs_1_2; lp_gibbs_1_3; lp_nuts_1_1])
#
# # l1 = layer(x=1:10000, y=lp_nuts_1_1, Geom.line)
# # l2 = layer(x=1:10000, y=lp_gibbs_1_1, Geom.line)
#
# plot(df_trace, x="Samples", y="lp", color="Engine", Geom.line)
