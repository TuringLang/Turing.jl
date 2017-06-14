# Helper functions used by tarce.jl and varinfo.jl

using Turing

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

using Turing: Sampler
