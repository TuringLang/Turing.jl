using Distributions
using Turing
using HDF5, JLD
using StatsFuns

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
    # Trapezoidal integration rule
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
μ = [   0,    1,     2, 3.5, 4.25] + 0.5*collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)

N = 100000
K = 500
# chain_pg = sample(gmm_gen(p, μ, σ), PG(50, N))
# x_pg = map(x_arr -> x_arr[1], chain_pg[:x])

# chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(N, PG(50, 1, :z), NUTS(10, 1000, 0.65, :x)))

# chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(5, 1, :z), HMC(K-1, 0.2, 8, :x); thin=false))

chain_gibbs = load(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld")["chain"]
chain_nuts = load(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld")["chain"]

x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])

# save(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld", "chain", chain_gibbs)
# save(TPATH*"/nips-2017/gmm/gibbs-chain-0.5.jld", "chain", chain_gibbs)

# chain_nuts = sample(gmm_gen_marg(p, μ, σ), NUTS(N, 0.65))


x_nuts = map(x_arr -> x_arr[1], chain_nuts[:x])

# save(TPATH*"/nips-2017/gmm/nuts-chain-0.5.jld", "chain", chain_nuts)

using Gadfly, DataFrames
using Colors


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
