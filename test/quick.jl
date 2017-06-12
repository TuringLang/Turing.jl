using Turing

@model ldamodel_vec(K, V, M, N, w, doc, beta, alpha) = begin
  theta = Matrix{Real}(K, M)
  theta ~ [Dirichlet(alpha)]

  phi = Matrix{Real}(V, K)
  phi ~ [Dirichlet(beta)]

  phi_dot_theta = log(phi * theta)
  for n = 1:N
    Turing.acclogp!(vi, phi_dot_theta[w[n], doc[n]])
  end
end

include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")

setchunksize(100)

sample(ldamodel_vec(data=ldastandata[1]), HMC(2000, 0.025, 10))
