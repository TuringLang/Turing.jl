function update(varInfo, samples_history, space)
  for var in keys(varInfo)
    if var.sym in space
      dist = varInfo.dists[var]
      s = 0
      for samples in samples_history
        s += samples[var.sym]
      end
      s /= length(samples_history)
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
