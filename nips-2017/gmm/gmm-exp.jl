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
  nbin = 100
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

using Gadfly, Colors
using DataFrames: DataFrame
n = 3
colors = distinguishable_colors(n, LCHab[LCHab(70, 60, 240)],
                                transform=c -> deuteranopic(c, 0.5),
                                lchoices=Float64[65, 70, 75, 80],
                                cchoices=Float64[0, 50, 60, 70],
                                hchoices=linspace(0, 330, 24))

M = 5
p = [ 0.2,  0.2,   0.2, 0.2,  0.2]
μ = [   0,    1,     2, 3.5, 4.25] + 2.5*collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)

N = 1000

particle_nums = [5, 10]
nrows = length(particle_nums)
HMC_runs = [10, 100, 500, 1000] .+ 1
ncols = length(HMC_runs)

vs_all = []

for K = HMC_runs

var_pn_plots = []

for pn = particle_nums

println("Running Gibbs")
chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(10, 1, :z), NUTS(K-1, 500, 0.65, :x); thin=false))
#chain_gibbs = load(TPATH*"/nips-2017/gmm/gibbs-chain-2.5.jld")["chain"]
#save(TPATH*"/nips-2017/gmm/gibbs-chain-0.5.jld", "chain", chain_gibbs)
x_gibbs = map(x_arr -> x_arr[1], chain_gibbs[:x])

println("Running NUTS")
chain_nuts = sample(gmm_gen_marg(p, μ, σ), NUTS(N, 0.65))
#chain_nuts = load(TPATH*"/nips-2017/gmm/nuts-chain-2.5.jld")["chain"]
#save(TPATH*"/nips-2017/gmm/nuts-chain-0.5.jld", "chain", chain_nuts)
x_nuts = map(x_arr -> x_arr[1], chain_nuts[:x])

gibbs_layer = layer(x=x_gibbs, Geom.density, Theme(default_color=colors[1]))
nuts_layer = layer(x=x_nuts, Geom.density, Theme(default_color=colors[2]))
contour_layer = layer([make_norm_pdf(μ, σ)], -5, 20, Theme(default_color=colors[3]))

layers = [gibbs_layer, nuts_layer, contour_layer]
labels = ["Gibbs", "NUTS", "Exact"]

select = [1,2,3]
gmm_density = plot(layers..., Guide.manual_color_key("", labels, colors),
                   Coord.cartesian(xmin=-5, xmax=20, ymin=0, ymax=1.0), 
                   Guide.xlabel(nothing), Guide.ylabel("Density"), Guide.title("PG($pn,1)+NUTS($K,500,0.64)"))

push!(var_pn_plots, gmm_density)

end

vs = vstack(var_pn_plots...)
push!(vs_all, vs)

end

s_all = hstack(vs_all...)

Gadfly.push_theme(:default)

draw(PDF(TPATH*"/nips-2017/gmm/gmm-density-vary-np.pdf", (4*ncols)inch, (2*nrows)inch), s_all)

