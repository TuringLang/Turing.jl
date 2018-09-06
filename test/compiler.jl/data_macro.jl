using Turing, Test

# define a simple function
f(; x = 1) = 2*x

# call f() with kwarg values filled from data
r = @data(f(), Dict(:x => 2))

# test
@test r == f(x=2)

# define a more advanced function call
g(; x = 1, y = 2) = 2*x*y

# call g(x = 2) only with kwarg values filled that are not defined in the function call
r = @data(g(x = 2), Dict(:y => 2, :x => 1))

@test r == g(x = 2, y = 2)

# predefine the dictionary and call the function afterwards
data = Dict(:x => 2)

# call g(y = 2) as above but using a variable data for the dict
r = @data(g(y = 2), data)

@test r == g(x = 2, y = 2)
