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
  nbin = 128
  x, y = gen_hist(img, nbin)
  x, y./trapz(x,y)
end

using Gadfly, Colors
using DataFrames: DataFrame
n = 3
colors = distinguishable_colors(n, LCHab[LCHab(70, 60, 240)],
                                transform=c -> deuteranopic(c, 0.5),
                                lchoices=Float64[65, 70, 75, 80],
                                cchoices=Float64[0, 50, 60, 70],
                                hchoices=linspace(0, 330, 24))

make_norm_pdf(p, μ, σ) = x -> map(i -> pdf(UnivariateGMM2(μ, σ, Categorical(p)), i), x)

visualize(x_gibbs, x_nuts, μ, xmin=-5, xmax=20) = begin
    x, y_g = make_vec(x_gibbs)
    gibbs_layer = layer(x=x, y=y_g, Geom.bar, Theme(default_color=colors[1]))
    x, y_n = make_vec(x_nuts)
    nuts_layer = layer(x=x, y=y_n, Geom.bar, Theme(default_color=colors[2]))
    contour_layer = layer([make_norm_pdf(p, μ, σ)], xmin, xmax, Theme(default_color=colors[3]))

    layers = [gibbs_layer, nuts_layer, contour_layer]
    labels = ["Gibbs", "NUTS", "Exact"]

    order = [3,1]
    plot_g = plot(layers[order]..., Guide.manual_color_key("", labels[order], colors[order]),
                 Coord.cartesian(xmin=xmin, xmax=xmax, ymin=0, ymax=1.0),
                 Guide.xlabel(nothing), Guide.ylabel("Density"), Guide.title("NUTS v.s. Gibbs"))
    
    order = [3,2]
    plot_n = plot(layers[order]..., Guide.manual_color_key("", labels[order], colors[order]),
                 Coord.cartesian(xmin=xmin, xmax=xmax, ymin=0, ymax=1.0),
                 Guide.xlabel(nothing), Guide.ylabel("Density"), Guide.title("NUTS v.s. Gibbs"))

    draw(PNG(15cm, 10cm), vstack(plot_g, plot_n))
end
