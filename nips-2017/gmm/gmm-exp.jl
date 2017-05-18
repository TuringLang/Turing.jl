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
    #= correction -h^2/12 * (f'(b) - f'(a))
    ha = x[2] - x[1]
    he = x[end] - x[end-1]
    ra = (y[2] - y[1]) / ha
    re = (y[end] - y[end-1]) / he
    r/2 - ha*he/12 * (re - ra)
    =#
    return r/2
end

make_vec(img::Vector) = begin
  nbin = 64
  x, y = gen_hist(img, nbin)
  x, y./trapz(x,y)
end

"
model {
// priors
  theta ~ dirichlet(alpha0_vec) ;
  for (k in 1:K) {
    mu[k] ~ normal (0.0, 1.0) ;
    sigma[k] ~ lognormal(0.0, 1.0) ;
  }
// likelihood
  for (n in 1:N) {
    real ps[K] ;
    for (k in 1 :K) {
      ps[k] <- log(theta[k]) + normal_log(y[n], mu[k], sigma[k]) ;
    }
    increment_log_prob(log_sum_exp(ps)) ;
  }
}
"

@model gmm(M, N, p, μ, σ) = begin
  z = tzeros(Int, N)
  x = tzeros(Real, N)
  for i = 1:N
    z[i] ~ Categorical(p)
    x[i] ~ Normal(μ[z[i]], σ[z[i]])
  end
end

@model gmm_gen(p, μ, σ) = begin
  z ~ Categorical(p)
  x ~ Normal(μ[z], σ[z])
end

make_norm_pdf(μ, σ) =
  x -> (pdf(Normal(μ[1], σ[1]), x) + pdf(Normal(μ[2], σ[2]), x) +
        pdf(Normal(μ[3], σ[3]), x) + pdf(Normal(μ[4], σ[4]), x) +
        pdf(Normal(μ[5], σ[5]), x)) / 5

vn = Turing.VarName(gensym(), :x, "", 0)
@model gmm_gen_marg(p, μ, σ) = begin
  if isempty(vi)
    Turing.push!(vi, vn, 0, Normal(0,1), 0)
    x = rand(Uniform(-20,20))
  else
    x = vi[vn]
  end
  Turing.acclogp!(vi, log(make_norm_pdf(μ, σ)(x)))
end

M = 5
p = [ 0.2,  0.2,   0.2, 0.2,  0.2]
μ = [   0,    1,     2, 3.5, 4.25] + 2.5*collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)

N = 100000
# chain_pg = sample(gmm_gen(p, μ, σ), PG(50, N))
# x_pg = map(x_arr -> x_arr[1], chain_pg[:x])

# chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(N, PG(50, 1, :z), NUTS(10, 1000, 0.65, :x)))
K = 500
# chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(5, 1, :z), HMC(K-1, 0.2, 8, :x); thin=false))
chain_nuts = load(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld")["chain"]
x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])

# save(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld", "chain", chain_gibbs)

# chain_nuts = sample(gmm_gen_marg(p, μ, σ), NUTS(N, 0.65))
chain_nuts = load(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld")["chain"]
x_nuts = map(x_arr -> x_arr[1], chain_nuts[:x])

# save(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld", "chain", chain_nuts)

using Gadfly

colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d", colorant"#c0392b"]

# plot_type = Geom.histogram(density=true)
plot_type = Geom.density
# pg_layer = layer(x=x_pg, plot_type, Theme(default_color=colors[1]))
gibbs_layer = layer(x=x_gibbs, plot_type, Theme(default_color=colors[2]))
nuts_layer = layer(x=x_nuts, plot_type, Theme(default_color=colors[3]))

# plot(pg_layer)
plot(gibbs_layer)
plot(nuts_layer)

contour_layer = layer([make_norm_pdf(μ, σ)], -5, 20, Theme(default_color=colors[4]))

plot(contour_layer)
# plot(gibbs_layer, contour_layer)

# layers = [pg_layer, gibbs_layer, nuts_layer, contour_layer]
layers = [gibbs_layer, nuts_layer, contour_layer]
# labels = ["PG", "Gibbs", "NUTS", "Exact"]
labels = ["Gibbs", "NUTS", "Exact"]

select = [1,2,3]
gmm_density = plot(layers[select]..., Guide.manual_color_key("", labels[select], colors[select+1]))






x_gibbs_x, x_gibbs_c = make_vec(x_gibbs)
gibbs_hist = layer(x=x_gibbs_x,y=x_gibbs_c, Geom.bar,  Theme(default_color=colors[1]))

x_nuts_x, x_nuts_c = make_vec(x_nuts)
nuts_hist = layer(x=x_nuts_x,y=x_nuts_c, Geom.bar,  Theme(default_color=colors[3]))

gmm_hist = plot(contour_layer, gibbs_hist, nuts_hist, Guide.xlabel("Value of x"), Guide.ylabel("Density"), Guide.manual_color_key("", labels[select], colors[[1,3,4]]))


# gibbs_trace = layer(x=1:length(chain_gibbs[:x]),y=chain_gibbs[:x],Geom.line,Theme(default_color=colors[2]))
# nuts_trace = layer(x=1:N,y=chain_nuts[:x],Geom.line, Theme(default_color=colors[3]))
# plot(gibbs_trace,nuts_trace,Guide.manual_color_key("Legend", labels[[2,3]], colors[[2,3]]))

Gadfly.push_theme(:default)

draw(PDF(TPATH*"/nips-2017/gmm/gmm-density.pdf", 8inch, 4.5inch), gmm_density)
draw(PDF(TPATH*"/nips-2017/gmm/gmm-hist.pdf", (16/3)inch, 3inch), gmm_hist)
