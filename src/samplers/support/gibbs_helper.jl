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
    val = vi[uid]
    if istrans(vi, uid)
      dist = getdist(vi, uid)
      val = reconstruct(dist, val)
      val = invlink(dist, val)
      val = Any[realpart(val)]
      val = length(val) == 1 ? val[1] : val   # Remove un-necessary []'s
    end
    samples[sym(uid)] = val
  end
  samples
end
