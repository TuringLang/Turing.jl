function update(varInfo, samples, space)
  vars = collect(keys(varInfo))
  syms = keys(samples.value)
  for sym in syms
    if sym in space
      if isa(samples.value[sym], Real)
        var = filter(v -> v.uid == Symbol("$sym"), vars)[1]
        dist = varInfo.dists[var]
        s = samples.value[sym]
        v = link(dist, s)
        val = vectorize(dist, v)
        varInfo.values[var] = val
      else isa(samples.value[sym], Array)
        s = samples.value[sym]
        for i = 1:length(s)
          var = filter(v -> v.uid == Symbol("$sym[$i]"), vars)[1]
          dist = varInfo.dists[var]
          v = link(dist, s[i])
          val = vectorize(dist, v)
          varInfo.values[var] = val
        end
      end
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
    if ~(var.sym in keys(samples))
      samples[var.sym] = Any[realpart(val)]
    else
      push!(samples[var.sym], realpart(val))
    end
  end
  # Remove un-necessary []'s
  for k in keys(samples)
    if isa(samples[k], Array) && length(samples[k]) == 1
      samples[k] = samples[k][1]
    end
  end
  samples
end
