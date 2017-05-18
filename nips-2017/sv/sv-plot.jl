using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

# chain = load("/home/kai/bak/sv-gibbs-pg-hmc.jld")["chain"]
chain_gibbs = load(TPATH*"/nips-2017/sv/chain-gibbs.jld")["chain"]
chain_nuts = load(TPATH*"/nips-2017/sv/chain-nuts.jld")["chain"]

using Gadfly

spl_colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d", colorant"#c0392b"]
lps_gibbs = chain_gibbs[:lp]

l1 = layer(x=15:1000, y=-lps_gibbs[15:end], Geom.line, Geom.line,Theme(default_color=spl_colors[1]))

lps_nuts = chain_nuts[:lp]

l2 = layer(x=15:1000, y=-lps_nuts[15:end], Geom.line, Geom.line,Theme(default_color=spl_colors[2]))

lp_plot = plot(l1, l2,
Guide.xlabel("Number of iterations"), Guide.ylabel("Negative log-posterior"),
Guide.title("Negative Log-posterior for the Stochastic Volatility Model"), Guide.manual_color_key("Legend", ["Gibbs", "NUTS"], spl_colors[1:2]))

draw(PDF(TPATH*"/nips-2017/sv/lp_plot.pdf", 8inch, 4.5inch), lp_plot)

describe(chain_gibbs)
describe(chain_nuts)
