using Turing
using Turing: checkindex, setval!, updategid!, vectorize

randr(vi::VarInfo, vn::VarName, dist::Distribution, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
    r
  else
    if count checkindex(vn, vi) end
    r = vi[vn]
    vi.logp += logpdf(dist, r, istransformed(vi, vn))
    r
  end
end

randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    push!(vi, vn, r, dist, spl.alg.gid)
    spl.info[:ranges_updated] = false
    r
  elseif isnan(vi, vn)
    r = rand(dist)
    setval!(vi, vectorize(dist, r), vn)
    r
  else
    if count checkindex(vn, vi, spl) end
    updategid!(vi, vn, spl)
    vi[vn]
  end
end
