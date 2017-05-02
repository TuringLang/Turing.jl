using Turing
using Turing: newvar!, checkindex, setval!, updategid!

randr(vi::VarInfo, vn::VarName, dist::Distribution, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    newvar!(vi, vn, dist)
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
    newvar!(vi, vn, dist, spl.alg.gid)
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
