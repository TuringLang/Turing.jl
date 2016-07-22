# Demonstration of Automatic Differentiation in Julia

# Definiton of Dual
type Dual
  a   ::    Real
  b   ::    Real
  function Dual(a::Real, b::Real)
    new(a, b)
  end
end

# Coversion between Dual and Real type
Base.convert(::Type{Dual}, x::Real) = Dual(x, 0)

# Overload operators according to the property of dual numbers
+(x::Dual, y::Dual) = Dual(x.a + y.a, x.b + y.b)
-(x::Dual) = Dual(-x.a, -x.b)
*(x::Dual, y::Dual) = Dual(x.a * y.a, x.b * y.a + x.a * y.b)
/(x::Dual, y::Dual) = Dual(x.a / y.b, (x.b * y.a - x.a * y.b) / y.a^2)

# Custom function applying to Dual
function Base.exp(x::Dual)
  Dual(exp(x.a), exp(x.a) * x.b)
end

# A simple case
x = Dual(1, 1)
y = Dual(1, 0)
z = Dual(3, 0)
(y + exp(x) * x * x) * z
