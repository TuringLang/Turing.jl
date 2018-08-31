# Helper function for numerical tests
check_numerical(chain, symbols::Vector{Symbol}, exact_vals::Vector;
                eps=0.2) = begin
  for (sym, val) in zip(symbols, exact_vals)
    E = mean(chain[sym])
    print("  $sym = $E ≈ $val (eps = $eps) ?")
    cmp = abs.(sum(mean(chain[sym]) - val)) <= eps
    if cmp
      printstyled("./\n", color = :green)
      printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :green)
    else
      printstyled(" X\n", color = :red)
      printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :red)
    end
  end
end
