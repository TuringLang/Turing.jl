using Gadfly
using DataFrames

doc"""
Function for visualization topic models.

Usage:

    vis_topic_res(samples, K, V, avg_range)

- `samples` is the chain return by `sample()`
- `K` is the number of topics
- `V` is the size of vocabulary
- `avg_range` is the end point of the running average
"""
vis_topic_res(samples, K, V, avg_range) = begin
    phiarr = mean(samples[:phi][1:avg_range])

    phi = Matrix(0, V)
    for k = 1:K
        phi = vcat(phi, phiarr[k]')
    end

    df = DataFrame(Topic = vec(repmat(collect(1:K)', V, 1)),
                  Word = vec(repmat(collect(1:V)', 1, K)),
                  Probability = vec(phi))

    p = plot(df, x=:Word, y=:Topic, color=:Probability, Geom.rectbin)

    p
end
