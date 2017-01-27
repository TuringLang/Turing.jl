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
