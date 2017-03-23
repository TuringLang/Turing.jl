function update(vi, samples, space)
  uids = collect(keys(vi))
  syms = keys(samples.value)
  for sym in syms
    if sym in space
      if isa(samples.value[sym], Real)
        uid = filter(uid -> uid == "$sym", uids)[1]
        dist = getdist(vi, uid)
        s = samples.value[sym]
        val = vectorize(dist, link(dist, s))
        vi[uid] = val
      else isa(samples.value[sym], Array)
        s = samples.value[sym]
        for i = 1:length(s)
          uid = filter(uid -> uid == "$sym[$i]", uids)[1]
          dist = getdist(vi, uid)
          val = vectorize(dist, link(dist, s[i]))
          vi[uid] = val
        end
      end
    end
  end
  vi
end

function varInfo2samples(vi)
  samples = Dict{Symbol, Any}()
  for uid in keys(vi)
    dist = getdist(vi, uid)
    val = vi[uid]
    val = reconstruct(dist, val)
    val = invlink(dist, val)
    sym = getsym(vi, uid)
    if ~(sym in keys(samples))
      samples[sym] = Any[realpart(val)]
    else
      push!(samples[sym], realpart(val))
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

function merge(this_vi, trace)
  that_vi = trace.vi
  idx = length(this_vi.idcs)
  for uid = keys(that_vi)
    val = getval(that_vi, uid)
    val = isa(val, TArray) ? Array(get(trace.task, val)) : val
    # val = isa(val, TArray) ? Vector(val) : val
    if ~haskey(this_vi, uid)
      sym = getsym(that_vi, uid)
      dist = getdist(that_vi, uid)
      addvar!(this_vi, uid, deepcopy(val), sym, dist)
    else
      setval!(this_vi, val, uid)
    end
  end
  this_vi
end
