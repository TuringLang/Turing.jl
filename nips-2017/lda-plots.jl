############
# Plotting #
############

using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

include(TPATH*"/nips-2017/"*"lda-settings.jl")

N = 3
spls = spls[1:N]
spls_un = spls_un[1:N]
spl_colors = spl_colors[1:N]

for iscollapsed = [true,false]

  layers = []
  for i = 1:N # N-1 here excludes PG

    chain = iscollapsed ?
            load(TPATH*"/nips-2017/lda-exps-chain-$i.jld")["chain"] :
            load(TPATH*"/nips-2017/lda-exps-chain-$i-un.jld")["chain"]

    lps = chain[:lp]

    l = layer(x = 1:length(lps), y = -lps, Geom.line,Theme(default_color=spl_colors[i]))

    push!(layers, l)
  end

  LDA_name = iscollapsed ? "Collapsed LDA" : "LDA"

  p = plot(layers..., #Scale.x_log10,
           Guide.xlabel("Number of iterations"), Guide.ylabel("Negative log-posterior"),
           Guide.title("Negative Log-posterior for the $LDA_name Model"),
           Guide.manual_color_key("Legend", iscollapsed ? spls : spls_un, spl_colors))

  if iscollapsed
    draw(PNG(TPATH*"/nips-2017/lda-exps-plt.png", 8inch, 4.5inch), p)
  else
    draw(PNG(TPATH*"/nips-2017/lda-exps-plt-un.png", 8inch, 4.5inch), p)
  end
end
