using StatsBase, Turing
using Gadfly

# include("ASCIIPlot.jl")
import Gadfly.ElementOrFunction

# First add a method to the basic Gadfly.plot function for QQPair types (generated by Distributions.qqbuild())
Gadfly.plot(qq::QQPair, elements::ElementOrFunction...) = Gadfly.plot(x=qq.qx, y=qq.qy, Geom.point, Theme(highlight_width=0px), elements...)

# Now some shorthand functions
qqplot(x, y, elements::ElementOrFunction...) = Gadfly.plot(qqbuild(x, y), elements...)
qqnorm(x, elements::ElementOrFunction...) = qqplot(Normal(), x, Guide.xlabel("Theoretical Normal quantiles"), Guide.ylabel("Observed quantiles"), elements...)

NSamples = 5000

@model gdemo_fw() = begin
  # s ~ InverseGamma(2,3)
  s = 1
  m ~ Normal(0, sqrt(s))
  y ~ MvNormal([m; m; m], [sqrt(s) 0 0; 0 sqrt(s) 0; 0 0 sqrt(s)])
end

@model gdemo_bk(x) = begin
  # Backward Step 1: theta ~ theta | x
  # s ~ InverseGamma(2,3)
  s = 1
  m ~ Normal(0,sqrt(s))
  x ~ MvNormal([m; m; m], [sqrt(s) 0 0; 0 sqrt(s) 0; 0 0 sqrt(s)])
  # Backward Step 2: x ~ x | theta
  y ~ MvNormal([m; m; m], [sqrt(s) 0 0; 0 sqrt(s) 0; 0 0 sqrt(s)])
end

fw = PG(50, NSamples)
# bk = Gibbs(10, PG(10,10, :s, :y), HMC(1, 0.25, 5, :m));
bk = HMCDA(50, 0.65, 0.2);

s = sample(gdemo_fw(), fw);
# describe(s)

N = div(NSamples, 50)

x = [s[1,:y,:].value...]
s_bk = Array{Turing.Chain}(undef, N)

simple_logger = Base.CoreLogging.SimpleLogger(stderr, Base.CoreLogging.Debug)
Base.CoreLogging.with_logger(simple_logger) do
  global x, bk, s_bk
  i = 1
  while i <= N
    s_bk[i] = sample(gdemo_bk(x), bk);
    x = [s_bk[i][end, :y, :]...];
    i += 1
  end
end

s2 = vcat(s_bk...);
# describe(s2)


using UnicodePlots

qqm = qqbuild(s[:m], s2[:m])
show(scatterplot(qqm.qx, qqm.qy, title = "QQ plot for m", canvas = DotCanvas))
show(scatterplot(qqm.qx[51:end-50], qqm.qy[51:end-50], title = "QQ plot for m (removing first and last 50 quantiles):", canvas = DotCanvas))

qqm = qqbuild(s[:m], s2[:m])
X = qqm.qx
y = qqm.qy
slope = (1 / (transpose(X) * X)[1] * transpose(X) * y)[1]

print("  slopeₛ = $slope ≈ 1 (ϵ = 0.1)")
ans1 = abs(slope - 1.0) <= 0.1
if ans1
  printstyled(" ✓\n", color=:green)
else
  printstyled(" X\n", color=:red)
  printstyled("    slope = $slope, diff = $(slope - 1.0)\n", color=:red)
end

# qqs = qqbuild(s[:s], s2[:s])
