function normalize!(x)
  norm = sum(x)
  x /= norm
  return x
end

function align(x,y)
  if length(x) < length(y)
    z = zeros(y)
    z[1:length(x)] = x
    x = z
  elseif length(x) > length(y)
    z = zeros(x)
    z[1:length(y)] = y
    y = z
  end

  return (x,y)
end

function kl(p :: Categorical, q :: Categorical)
  a,b = align(p.p, q.p)
  return kl_divergence(a,b)
end

function kl(p::Normal, q::Normal)
  return (log(q.σ / p.σ) + (p.σ^2 + (p.μ - q.μ)^2) / (2 * q.σ^2) - 0.5)
end
