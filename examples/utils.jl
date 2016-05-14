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

immutable PolyaUrn
  counts::TArray{Float64, 1}
  alpha::Float64
end

PolyaUrn(alpha) = PolyaUrn(convert(TArray, [alpha]),  alpha)

function randclass(urn::PolyaUrn)
  counts = localcopy(urn.counts)
  weights = counts ./ sum(counts)
  @assume c ~ Categorical(weights)
  if c == length(counts)
    urn.counts[end] = 1
    push!(urn.counts, urn.alpha)
  elseif c < length(counts) && c > 0
    urn.counts[c] += 1
  else
    #println(weights)
    #println("before assume: $(previous_db)")
    #println("after assume: $(current_randomdb())")
    error("RANDCLASS: class id: $c, urn.counts: $(urn.counts), randomdb: $(current_randomdb())")
  end
  return Int64(c)::Int64
end

Base.deepcopy(urn::PolyaUrn) = PolyaUrn(deepcopy(urn.counts), urn.alpha)
