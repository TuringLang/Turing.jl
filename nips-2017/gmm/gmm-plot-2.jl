using Turing
using HDF5, JLD
using Mamba: summarystats

TPATH = Pkg.dir("Turing")

gibbs_chains = [load(TPATH*"/nips-2017/gmm/gibbs-chain-$s.jld")["chain"] for s in ["0.5", "2.5"]]
nuts_chains = [load(TPATH*"/nips-2017/gmm/nuts-chain-$s.jld")["chain"] for s in ["0.5", "2.5"]]


gibbs_smrs = map(c -> summarystats(c), gibbs_chains)
nuts_smrs = map(c -> summarystats(c), nuts_chains)

gibbs_ess = map(s -> s.value[4,5,1], gibbs_smrs)
nuts_ess = map(s -> s.value[3,5,1], nuts_smrs)

gibbs_time = map(c -> sum(c[:elapsed]), gibbs_chains)
nuts_time = map(c -> sum(c[:elapsed]), nuts_chains)

using Gadfly, DataFrames
using Colors

df = DataFrame(Run = [1,2,1,2], Engine = ["Gibbs", "Gibbs", "NUTS", "NUTS"], Time = [gibbs_time..., nuts_time...], ESS = [gibbs_ess..., nuts_ess...])

# plot(df, x="Engine", y="ESS", Geom.bar)
# plot(df, x="Engine", y="Time", Geom.bar)

p_ess = plot(df, xgroup="Run", x="Engine", y="ESS", Geom.subplot_grid(Geom.bar, Guide.title(nothing),Guide.xlabel(nothing), ),Guide.xlabel(nothing))
p_time = plot(df, xgroup="Run", x="Engine", y="Time", Geom.subplot_grid(Geom.bar,Guide.xlabel(nothing)),Guide.xlabel(nothing),)

p_stack = vstack(p_ess,p_time)

Gadfly.push_theme(:default)
draw(PDF(TPATH*"/nips-2017/gmm/gmm-ess-time.pdf", 4inch, 4inch), p_stack)


x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])


n = 3
colors = distinguishable_colors(n, LCHab[LCHab(70, 60, 240)],
                       transform=c -> deuteranopic(c, 0.5),
                       lchoices=Float64[65, 70, 75, 80],
                       cchoices=Float64[0, 50, 60, 70],
                       hchoices=linspace(0, 330, 24))

labels = ["Gibbs", "NUTS", "Exact"]

contour_layer = layer([make_norm_pdf(μ, σ)], -5, 10, Theme(default_color=colors[3]))

x_gibbs_x, x_gibbs_c = make_vec(x_gibbs)
gibbs_hist = layer(x=x_gibbs_x,y=x_gibbs_c, Geom.bar,  Theme(default_color=colors[1]))

x_nuts_x, x_nuts_c = make_vec(x_nuts)
nuts_hist = layer(x=x_nuts_x,y=x_nuts_c, Geom.bar,  Theme(default_color=colors[2]))

gmm_hist = plot(contour_layer, gibbs_hist, nuts_hist,
                Guide.xlabel("Value of x"), Guide.ylabel("Density"),
                Guide.manual_color_key("", labels[[2,1,3]], colors[[2,1,3]]),
                Theme(major_label_font_size=9pt),Coord.cartesian(xmin=-5, xmax=10, ymin=0, ymax=0.8))





# hstack(p3,p4)

Gadfly.push_theme(:default)
draw(PDF(TPATH*"/nips-2017/gmm/gmm-hist.pdf", 8inch, 2inch), gmm_hist)
