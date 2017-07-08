@model school8(J, y, sigma) = begin
  mu ~ NoInfo()
  tau ~ NoInfoPos(0)
  eta = Vector{Real}(J)
  eta ~ [Normal(0, 1)]
  y ~ MvNormal(mu .+ tau .* eta, sigma)
end
