############
# Plotting #
############

TPATH = Pkg.dir("Turing")

include(TPATH*"/nips-2017/"*"lda-settings.jl")

for iscollapsed = [true,false]

  layers = []
  for i = 1:N
    lps = iscollapsed ?
          readdlm(TPATH*"/nips-2017/lda-exps-lp-$i.txt") :
          readdlm(TPATH*"/nips-2017/lda-exps-lp-$i-un.txt")

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
