using Distributions
using Turing

# 250-dimensional multivariate normal
@model mvn_nuts(A) = begin
  dim, _ = size(A)
  Θ ~ MvNormal(zeros(dim), A)
end

# Sample a precision matrix A from a Wishart distribution
# with identity scale matrix and 250 degrees of freedome
A = rand(Wishart(250, eye(250)))

chain = sample(mvn_nuts(A), NUTS(1000, 0.65))
