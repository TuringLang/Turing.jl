# Single line comments

#=
  Multiple
  line
  comments
=#

##########################################
## 1. Primitive Datatypes and Operators ##
##########################################

3
typeof(3)
3.2
typeof(3.2)
2 + 1im
typeof(2 + 1im)
2//3
typeof(2//3)

1 + 1
3.2 - 1.1
(2 + 1im) * (1 + 1im)
35 / 5
5 / 2         # Int / Int => Float (always)
div(5, 2)     # div(Int, Int) => Int (truncated)
5 \ 35
2 ^ 10
12 % 10

(1 + 3) * 2

~2            #=
                bitwise not
                2 = 010 in 2's complement
                ~2 = 101
                101 in 2's complement is -3
              =#
3 & 5         # 0011 AND 1001 = 0001
2 | 4
2 $ 4         # bitwise xor
2 >>> 1       # logical shift right
2 >> 1        # arithmetic shift right
2 << 1        # logical/arithmetic shift left

bits(2)
bits(-3)
bits(2.0)

true
false

!true
!false
1 == 1
2 == 1
1 != 1
2 != 1
1 < 10
1 > 10
2 <= 2
2 >= 2

1 < 2 < 3       # comparisons can be chained
2 < 3 < 2

"This is a string."
'c'
"This is a string."[1]
"2 + 2 = $(2 + 2)"
@printf "%d is less than %f\n" 4.5 5.3

# available number format characters are f, e, g, c, s, p, d:
# (pi is a predefined constant; however, since its type is
# "MathConst" it has to be converted to a float to be formatted)
@printf "fix trailing precision: %0.3f\n" float(pi)
#> fix trailing precision: 3.142
@printf "scientific form: %0.6e\n" 1000pi
#> scientific form: 3.141593e+03
# g is not implemented yet
@printf "a character: %c\n" 'α'
#> a character: α
@printf "a string: %s\n" "look I'm a string!"
#> a string: look I'm a string!
@printf "right justify a string: %50s\n" "width 50, text right justified!"
#> right justify a string:                    width 50, text right justified!
@printf "a pointer: %p\n" 100000000
#> a pointer: 0x0000000005f5e100
@printf "print a integer: %d\n" 1e10
#> print an integer: 10000000000

println("I'm Julia. Nice to meet you!")

"good" > "bye"  # lexicographical comparisons
"good" == "good"
"1 + 2 = 3" == "1 + 2 = $(1 + 2)"

c1 = 'a'
println(c1, " accii value = ", Int(c1))

s1 = "I love Freyα."
s1_caps = uppercase(s1)
s1_lower = lowercase(s1)

# show prints the raw value
show(s1[11]); println()
show(s1[1:3]); println()

# strings can also be concatenated using the * operator
s2 = "this" * " and" * " that"
s3 = string("this", " and", " that")
s4 = "hello " ^ 3

s5 = "The quick brown fox jumps over the lazy dog α,β,γ"
i = search(s5, 'o')
i = search(s5, "fox")
r = replace(s5, "brown", "red")
r = search(s5, r"b[\w]*n")
r = replace(s5, r"b[\w]*n", "red")
# first match
r = match(r"b[\w]*n", s5)
# each match
r = matchall(r"[\w]{4,}", s5)
for(i in r) print("\"$i\" ") end

# the strip function works the same as python:
# e.g., with one argument it strips the outer whitespace
r = strip("hello ")
# or with a second argument of an array of chars it strips any of them;
r = strip("hello ", ['h', ' '])

r = split("hello, there,bob", ',')
r = split("hello, there,bob", ", ")
r = split("hello, there,bob", [',', ' '], limit=0, keep=false)
# the last two arguements are limit and include_empty, see docs

r = join(collect(1:10), ", ")

##################################
## 2. Variables and Collections ##
##################################

some_var = 5
some_var

try
  some_other_var
catch e
  println(e)
end

SomeOtherVar123! = 6
α = 0.1
2 * π

#=
  1. Names of Types begin with a capital letter and word separation is shown with CamelCase instead of underscores.
  2. Names of functions and macros are in lower case, without underscores.
  3. Functions that modify their inputs have names that end in !. These functions are sometimes called mutating functions or in-place functions.
=#

a = Int64[]             # Vector of Int64
a = (Array{Int64, 1})[] # Vecotr of Array of Int64

b = [4, 5, 6]
b = [4; 5; 6]
b[1]            # index starts with 1

matrix = [1 2 2; 3 4 4; 5 6 6]

b = Int64[4, 5, 6]
push!(a, 1)
push!(a, 2)
push!(a, 4)
push!(a, 3)
append!(a, b)

pop!(b)
b

a[1]
a[end]
shift!(a)       # pop the leftmost element
unshift!(a, 7)  # append to the left

a = collect(1:4)
aRepeat = repeat(a, inner=[2], outer=[1])
aRepeat = repeat(a, inner=[1], outer=[2])

arr = [5, 4, 6]
sort(arr)
arr
sort!(arr)
arr

try
  a[0]
  a[end+1]
catch e
  println(e)
end

a = [1:5;]        # Vector
a = collect(1:5)  # Vector
a = 1:5           # UnitRange
a[1:3]
a[2:end]

arr = [3, 4, 5]
splice!(arr, 2)
arr

b = [1, 2, 3]
append!(a, b)
a

in(1, a)
in('1', a)

length(a)

m1 = hcat(repeat([1, 2], inner=[1], outer=[3*2]),
          repeat([1, 2, 3], inner=[2], outer=[2]),
          repeat([1, 2, 3, 4], inner=[3], outer=[1]))
m2 = repmat(m1, 1, 2)
m3 = repmat(m1, 2, 1)
m4 = [i+j for i=1:2, j=1:3]
m5 = ASCIIString["Hi Im element # $(i+2*(j-1 + 3*(k-1)))" for i=1:2, j=1:3, k=1:2]

sum(m4, 2)
sum(m4, (1, 2))

maximum(m4, 1)
findmax(m4, 2)

m4 .+ 3       # add 3 to all elements
m4 .+ [1, 2]  # adds vector [1,2] to all elements along first dim
m4[:,1]

m5 = [i+j+k for i=1:2, j=1:3, k=1:2]
squeeze(m5[:, 2, :], 2)

m5[:,:,1] = rand(1:6, 2, 3)

tup = (1, 2, 3)
tup[1]
try:
  tup[1] = 3
catch e
  println(e)
end

length(tup)
tup[1:2]
in(2, tup)

# unpack tuple into varibales
a, b, c = tup
# tuples are created even if you leave out the parentheses
d, e, f = 4, 5, 6

(1,) == 1
(1) == 1

e, d = d, e
e, d

empty_dict = Dict()

filled_dict = Dict("one" => 1, "two" => 2)
filled_dict["one"]

keys(filled_dict)
values(filled_dict)

in(("one" => 1), filled_dict)
in(("two" => 3), filled_dict)
haskey(filled_dict, "one")
haskey(filled_dict, 1)

try
  filled_dict["three"]
catch e
  println(e)
end

get(filled_dict, "one", 4)    # get method can avoid the error
get(filled_dict, "four", 4)   # by providing a default value

empty_set = Set()
filled_set = Set([1, 2, 2, 3, 4])

push!(filled_set, 5)

in(2, filled_set)
in(10, filled_set)

other_set = Set([3, 4, 5, 6])
intersect(filled_set, other_set)
union(filled_set, other_set)

#####################
## 3. Control Flow ##
#####################

some_var = 5

if some_var > 10
  println("some_var is totally bigger than 10.")
elseif some_var < 10
  println("some_var is smaller than 10.")
else
  println("some_var is indeed 10.")
end

for animal = ["dog", "cat", "mouse"]
  println("$animal is a mammal")
end

for animal in ["dog", "cat", "mouse"]
  println("$animal is a mammal")
end

for a in Dict("dog"=>"mammal","cat"=>"mammal","mouse"=>"mammal")
    println("$(a[1]) is a $(a[2])")
end

for (k,v) in Dict("dog"=>"mammal","cat"=>"mammal","mouse"=>"mammal")
    println("$k is a $v")
end

x = 0
while x < 4
  println(x)
  x += 1
end

try
   error("self-defined error")
catch e
   println("caught it $e")
end

##################
## 4. Functions ##
##################

function add(x, y)
  println("x is $x and y is $y")
  # functions return the value of their last statement
  x + y
end

add(5, 6)

# arguments types can be defined in function definition
function quadratic2(a::Float64, b::Float64, c::Float64)
  quadratic(a, sqr_term, b) = (-b + sqr_term) / 2a
  sqr_term = sqrt(b ^ 2 - 4a * c)
  r1 = quadratic(a, sqr_term, b)
  r2 = quadratic(a, -sqr_term, b)
  r1, r2
end

quad1, quad2 = quadratic2(1.0, -2.0, 1.0)

# compact assignment of functions
f_add(x, y) = x + y
f_add(3, 4)

# function can also return multiple values as tuple
fun(x, y) = x + y, x - y
fun(3, 4)

function varargs(args...)
  return args
end

#=
  The ... is called a splat.
  We just used it in a function definition.
  It can also be used in a function call,
  where it will splat an Array or Tuple's contents
  into the argument list.
=#

varargs(1, 2, 3)

add([5, 6]...)    # this is equvalent to add(5, 6)
x = (5, 6)
add(x...)

function defaults(a, b, x=5, y=6)
  return "$a $b and $x $y"
end

defaults('h', 'g')
defaults('h', 'g', 'j')
defaults('h', 'g', 'j', 'k')

try
    defaults('h')
catch e
    println(e)
end

# define functions that take keyword arguments
function keyword_args(;k1=4, name2="hello") # note the ;
    return ["k1"=>k1, "name2"=>name2]
end

keyword_args(name2="ness")
keyword_args(k1="mine")
keyword_args()

function all_the_args(normal_arg, optional_positional_arg=2; keyword_arg="foo")
    println("normal arg: $normal_arg")
    println("optional arg: $optional_positional_arg")
    println("keyword arg: $keyword_arg")
end

all_the_args(1, 3, keyword_arg=4)

function create_adder(x)
    adder = function (y)
        return x + y
    end
    return adder
end

add_1 = create_adder(1)   # adder = λ y: 1 + y
add_1(2)                  # r_1eturn : 1 + 2

# "stabby lambda syntax" for creating anonymous functions
(x -> x > 2)(3)
(x -> x ^ 2)(2)

# another implementation of create_adder
function create_adder(x)
    y -> x + y
end

adder = create_adder(1)
adder = y -> 1 + y

# or name the internal function
function create_adder(x)
    function adder(y)
        x + y
    end
    adder
end

add_10 = create_adder(10)
add_10(3)

# There are built-in higher order functions
map(add_10, [1,2,3]) # => [11, 12, 13]
filter(x -> x > 5, [3, 4, 5, 6, 7]) # => [6, 7]

# map can also be done in list comprehensions
[add_10(i) for i=[1, 2, 3]] # => [11, 12, 13]
[add_10(i) for i in [1, 2, 3]] # => [11, 12, 13]

# passing mutiple parameters should be done in tuple
my_add = (x, y) -> x + y
reduce(my_add, [1, 2, 3, 4])

##############
## 5. Types ##
##############
typeof(5)

typeof(Int64)
typeof(DataType)  # # DataType is the type that represents types, including itself.

# type Name
#   field::OptionalType
#   ...
# end
type Tiger
  taillength::Float64
  coatcolor # not including a type annotation is the same as `::Any`
end

tigger = Tiger(3.5, "orange")

# The type doubles as the constructor function for values of that type
sherekhan = typeof(tigger)(5.6,"fire")

abstract Cat

subtypes(Number)
subtypes(Cat)
subtypes(DataType)
subtypes(AbstractString)

function loopSuperTypes(constant)
  superType = typeof(constant)
  println("typeof($constant) => $superType")
  while superType != Any
    childType = superType
    superType = super(childType)
    println("super($childType) => $superType")
  end
end

loopSuperTypes(5)
loopSuperTypes("Hello")

t1 = typeof(5)
t2 = super(t1)
t3 = super(t2)

# <: is the subtyping operator
type Lion <: Cat # Lion is a subtype of Cat
  mane_color
  roar::AbstractString
end

# define a constructor from its super
Lion(roar::AbstractString) = Lion("green", roar)

type Panther <: Cat # Panther is also a subtype of Cat
  eye_color
  Panther() = new("green")
  # Panthers will only have this constructor, and no default constructor.
end
# Using inner constructors, like Panther does, gives you control
# over how values of the type can be created.
# When possible, you should use outer constructors rather than inner ones.

##########################
## 6. Multiple-Dispatch ##
##########################

function meow(animal::Lion)
  animal.roar
end

function meow(animal::Panther)
  "grrr"
end

function meow(animal::Tiger)
  "rawwwr"
end

meow(tigger)
meow(Lion("brown","ROAAR"))
meow(Panther())

issubtype(Tiger, Cat)
issubtype(Lion, Cat)
issubtype(Panther, Cat)

function pet_cat(cat::Cat)
  println("The cat says $(meow(cat))")
end

pet_cat(Lion("42"))
try
  pet_cat(tigger)
catch e
  println(e)
end

function fight(t::Tiger, c::Cat)
  println("THe $(t.coatcolor) tiger wins!")
end

fight(tigger, Panther())
fight(tigger, Lion("ROAR"))

fight(t::Tiger, l::Lion) = println("The $(l.mane_color)-maned lion wins!")
fight(tigger,Panther())
fight(tigger,Lion("ROAR"))

fight(l::Lion, c::Cat) = println("The victorious cat says $(meow(c))")

fight(Lion("balooga!"),Panther())
try
  fight(Panther(), Lion("RAWR"))
catch e
  println(e)
end

fight(c::Cat,l::Lion) = println("The cat beats the Lion")
# WARNING: New definition
#     fight(Main.Cat, Main.Lion) at /Users/kai/fun/fun.jl:487
# is ambiguous with:
#     fight(Main.Lion, Main.Cat) at /Users/kai/fun/fun.jl:478.
# To fix, define
#     fight(Main.Lion, Main.Lion)
# before the new definition.

fight(Lion("RAR"),Lion("brown","rarrr"))

fight(l::Lion, l2::Lion) = println("The lions come to a tie")
fight(Lion("RAR"), Lion("brown","rarrr"))

square_area(l) = l * l
square_area(5)

code_native(square_area, (Int32,))
code_native(square_area, (Float32,))
code_native(square_area, (Float64,))

# Note that Julia will use floats if any of the arguments are floats
circle_area(r) = pi * r * r
circle_area(5)

code_native(circle_area, (Int32,))
code_native(circle_area, (Float32,))
code_native(circle_area, (Float64,))

#################
## 7. Plotting ##
#################

# The GadFly package is for plotting and the Cairo package is for saving images.
using Gadfly, Cairo

# Plotting arrays
plot(x=rand(10), y=rand(10))

plot(x=rand(10), y=rand(10), Geom.point, Geom.line)

myPlot = plot(x=1:10, y=2.^rand(10), Scale.y_sqrt, Geom.point, Geom.smooth, Guide.xlabel("Stimulus"), Guide.ylabel("Response"), Guide.title("Dog Training"))

draw(PNG("myPlot.png", 4inch, 3inch), myPlot)

# The RDatasets package collects example data sets from R packages.
using RDatasets

# Plotting data frames
plot(dataset("datasets", "iris"), x="SepalLength", y="SepalWidth", Geom.point)

plot(dataset("car", "SLID"), x="Wages", color="Language", Geom.histogram)

# Plotting functions and expressions
plot([sin, cos], 0, 25)

# Layers, labels and title
plot(layer(x=rand(10), y=rand(10), Geom.point, order=1), layer(x=rand(10), y=rand(10), Geom.line, order=2), Guide.XLabel("XLabel"), Guide.YLabel("YLabel"), Guide.Title("Title"));

# Stacking
p1 = plot(x=[1,2,3], y=[4,5,6])
p2 = plot(x=[1,2,3], y=[6,7,8])
draw(PDF("p1and2.pdf", 6inch, 6inch), vstack(p1,p2))
p3 = plot(x=[5,7,8], y=[8,9,10])
p4 = plot(x=[5,7,8], y=[10,11,12])
draw(PDF("p1to4.pdf", 6inch, 9inch), vstack(hstack(p1,p2),hstack(p3,p4)))

############
## 8. I/O ##
############

fName = "simple.dat"

open(fName, "r") do f
  for line in eachline(f)
    print(line)
  end
end

f = open(fName, "r")
readlines(f)
fString = readall(f)

outFile = "outfile.dat"
f = open(outFile, "w")
println(f, "some contents")
close(f)

###################
## 9. DataFrames ##
###################

using DataFrames
DataFrame(A = [1, 2], B = [e, pi], C = ["xx", "xy"])

iris = readtable("iris.csv")

using RDatasets
iris = dataset("datasets","iris")

iris[:SepalLength] = round(Integer, iris[:SepalLength])
iris[:SepalWidth] = round(Integer, iris[:SepalWidth])
tabulated = by(iris, [:Species, :SepalLength, :SepalWidth], df -> size(df, 1))

gdf = groupby(iris,[:Species, :SepalLength, :SepalWidth])

insert!(iris, 5, rand(nrow(iris)), :randCol)
delete!(iris, :randCol)
