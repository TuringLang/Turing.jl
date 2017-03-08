function update(varInfo, samples, space)
  for var in keys(varInfo)
    if var.sym in space
      dist = varInfo.dists[var]
      s = samples[var.sym]
      v = link(dist, s)
      val = vectorize(dist, v)
      varInfo.values[var] = val
    end
  end
  varInfo
end

function varInfo2samples(varInfo)
  samples = Dict{Symbol, Any}()
  for var in keys(varInfo)
    dist = varInfo.dists[var]
    val = varInfo[var]
    val = reconstruct(dist, val)
    val = invlink(dist, val)
    samples[var.sym] = val
  end
  samples
end
