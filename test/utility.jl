# Helper functions used by tarce.jl and varinfo.jl
# TODO: can we somehow update the tests so that we can remove these two functions below?

using Turing
using Turing: checkindex, setval!, updategid!, acclogp!, vectorize, CACHERESET, VarInfo, VarName, Sampler

randr(vi::VarInfo, vn::VarName, dist::Distribution, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
    r
  else
    if count checkindex(vn, vi) end
    r = vi[vn]
    acclogp!(vi, logpdf(dist, r, istrans(vi, vn)))
    r
  end
end
println("[test/utility.jl] randr() reloaded")

randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    push!(vi, vn, r, dist, spl.alg.gid)
    spl.info[:cache_updated] = CACHERESET   # sanity flag mask for getidcs and getranges
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
println("[test/utility.jl] randr() reloaded")

# Helper function for numerical tests

check_numerical(chain::Turing.Chain, symbols::Vector{Symbol}, exact_vals::Vector;
                eps=0.2) = begin
  for (sym, val) in zip(symbols, exact_vals)
    E = mean(chain[sym])
    print("  $sym = $E ≈ $val (ϵ = $eps) ?")
    cmp = abs(sum(mean(chain[sym]) - val)) <= eps
    if cmp
      print_with_color(:green, " ✓\n")
      print_with_color(:green, "    $sym = $E, diff = $(abs(E - val))\n")
    else
      print_with_color(:red, " X\n")
      print_with_color(:red, "    $sym = $E, diff = $(abs(E - val))\n")
    end
  end
end
