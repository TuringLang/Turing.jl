using Distributions, Turing, HDF5, JLD, StatsFuns

TPATH = Pkg.dir("Turing")

gen_hist(img::Vector, nbin::Int) = begin
  min_x = -5
  max_x = 20

  x = zeros(nbin)
  x[1] = min_x
  step = (max_x - min_x) / nbin
  for i = 2:nbin
    x[i] = x[i-1] + step
  end
  x += step / 2
  counts = zeros(Int, nbin)
  for val in img
    idx = ceil(Int, (val - min_x) / step)
    # idx = idx >= nbin ? nbin : idx
    if idx > 0 && idx <= nbin
      counts[idx] += 1
    end
  end

  x, counts
end

trapz{Tx<:Number, Ty<:Number}(x::Vector{Tx}, y::Vector{Ty}) = begin
    local n = length(x)
    if (length(y) != n)
        error("Vectors 'x', 'y' must be of same length")
    end
    r = zero(zero(Tx) + zero(Ty))
    if n == 1; return r; end
    for i in 2:n
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])
    end

    r/2
end

make_vec(img::Vector) = begin
  nbin = 100
  x, y = gen_hist(img, nbin)
  x, y./trapz(x,y)
end

make_norm_pdf(μ, σ) =
  x -> (pdf(Normal(μ[1], σ[1]), x) + pdf(Normal(μ[2], σ[2]), x) +
        pdf(Normal(μ[3], σ[3]), x) + pdf(Normal(μ[4], σ[4]), x) +
        pdf(Normal(μ[5], σ[5]), x)) / 5


M = 5
p = [ 0.2,  0.2,   0.2, 0.2,  0.2]
μ = [   0,    1,     2, 3.5, 4.25] + 2.5*collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)

N = 100000
K = 500

using Gadfly, DataFrames, Colors
Gadfly.push_theme(:default)

n = 3
colors = distinguishable_colors(n, LCHab[LCHab(70, 60, 240)],
                       transform=c -> deuteranopic(c, 0.5),
                       lchoices=Float64[65, 70, 75, 80],
                       cchoices=Float64[0, 50, 60, 70],
                       hchoices=linspace(0, 330, 24))





μ = [   0,    1,     2, 3.5, 4.25] + 2.5*collect(0:4)
# chain_gibbs_1 = load(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld")["chain"]
x_gibbs = map(x_arr -> x_arr[1], chain_gibbs_1[:x])

# chain_nuts_1 = load(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld")["chain"]
x_nuts = map(x_arr -> x_arr[1], chain_nuts_1[:x])

contour_layer = layer([make_norm_pdf(μ, σ)], -3, 17, Theme(default_color=colors[3]))

x_gibbs_x, x_gibbs_c = make_vec(x_gibbs)
gibbs_hist = layer(x=x_gibbs_x,y=x_gibbs_c, Geom.bar,  Theme(default_color=colors[1]))

x_nuts_x, x_nuts_c = make_vec(x_nuts)
nuts_hist = layer(x=x_nuts_x,y=x_nuts_c, Geom.bar,  Theme(default_color=colors[2]))

labels = ["Gibbs", "NUTS", "Exact"]
select = [1,2,3]
gmm_hist_nuts_fail = plot(contour_layer, nuts_hist, Guide.xticks(label=false), Guide.yticks(label=false, ticks=collect(0:0.2:1.2)),
                          Guide.xlabel(nothing), Guide.ylabel(nothing),
                          Guide.manual_color_key("", labels[[2,1,3]], colors[[2,1,3]]),
                          Theme(major_label_font_size=9pt),Coord.cartesian(xmin=-3, xmax=17, ymin=0, ymax=1.2))

gmm_hist_nuts_fail_2 = plot(contour_layer, gibbs_hist,
                            Guide.xlabel("Value of x"), Guide.ylabel(nothing),  Guide.yticks(label=false),
                            # Guide.manual_color_key("", labels[[1,3]], colors[[1,3]]),
                            Theme(major_label_font_size=9pt),Coord.cartesian(xmin=-3, xmax=19.75, ymin=0, ymax=0.7))

μ = [   0,    1,     2, 3.5, 4.25] + 0.5*collect(0:4)

# chain_gibbs = load(TPATH*"/nips-2017/gmm/gibbs-chain-0.5.jld")["chain"]
x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])

# chain_nuts = load(TPATH*"/nips-2017/gmm/nuts-chain-0.5.jld")["chain"]
x_nuts = map(x_arr -> x_arr[1], chain_nuts[:x])

contour_layer = layer([make_norm_pdf(μ, σ)], -5, 12, Theme(default_color=colors[3]))

x_gibbs_x, x_gibbs_c = make_vec(x_gibbs)
gibbs_hist = layer(x=x_gibbs_x,y=x_gibbs_c, Geom.bar,  Theme(default_color=colors[1]))

x_nuts_x, x_nuts_c = make_vec(x_nuts)
nuts_hist = layer(x=x_nuts_x,y=x_nuts_c, Geom.bar,  Theme(default_color=colors[2]))

gmm_hist_nuts_good = plot(contour_layer, nuts_hist, Guide.yticks(ticks=collect(0:0.2:1.2)),
                          Guide.xlabel(nothing), Guide.ylabel("Density"), Guide.xticks(label=false),
                          # Guide.manual_color_key("", labels[[2,1,3]], colors[[2,1,3]]),
                          Theme(major_label_font_size=9pt),Coord.cartesian(xmin=-5, xmax=12, ymin=0, ymax=1.2))


gmm_hist_nuts_good_2 = plot(contour_layer, gibbs_hist,
                            Guide.xlabel("Value of x"), Guide.ylabel("Density"),
                            # Guide.manual_color_key("", labels[[2,1,3]], colors[[2,1,3]]),
                            Theme(major_label_font_size=9pt),Coord.cartesian(xmin=-5, xmax=12, ymin=0, ymax=0.7))

# hs = hstack(gmm_hist_nuts_good, gmm_hist_nuts_fail)

gs = gridstack([gmm_hist_nuts_good gmm_hist_nuts_fail; gmm_hist_nuts_good_2 gmm_hist_nuts_fail_2])

draw(PDF(TPATH*"/nips-2017/gmm/gmm-hist.pdf", 8inch, 3inch), gs)










N = 50000
df_trace = DataFrame(Samples=[collect(1:N); collect(1:N)],
                     Engine=[["Gibbs" for _ = 1:N]..., ["NUTS" for _ = 1:N]...],
                     x=[x_gibbs; x_nuts])
trace_x = plot(df_trace, x="Samples", y="x", color="Engine", Geom.line, Guide.colorkey(""), Guide.xlabel(nothing))
# draw(PDF(TPATH*"/nips-2017/gmm/gmm-trace-x.pdf", 8inch, 2inch), trace_x)

# sv_nuts_chain = load(TPATH*"/nips-2017/sv/new-first/chain-nuts.jld")["chain"]
# sv_gibbs_chain = load(TPATH*"/nips-2017/sv/new-first/chain-gibbs.jld")["chain"]
sv_nuts_chain = load("/home/kai/sv-nuts-1-1.jld")["chain"]
sv_gibbs_chain = load("/home/kai/sv-gibbs-1-3.jld")["chain"]

sv_nuts_lps = sv_nuts_chain[:lp]
sv_gibbs_lps = sv_gibbs_chain[:lp]

N = 10000
SV_start = 10
df_trace = DataFrame(Samples=[collect(SV_start:N); collect(SV_start:N)],
                     Engine=[["Gibbs" for _ = SV_start:N]..., ["NUTS" for _ = SV_start:N]...],
                     lp=[sv_gibbs_lps[SV_start:N]; sv_nuts_lps[SV_start:N]])
trace_lp = plot(df_trace, x="Samples", y="lp", color="Engine", Geom.line, Guide.colorkey(""), Guide.ylabel("lp"), Guide.xlabel("Number of iterations"), Theme(major_label_font_size=9pt))

vs = vstack(trace_x,trace_lp)
draw(PDF(TPATH*"/nips-2017/trace-sv-gmm.pdf", 8inch, 3.5inch), vs)
