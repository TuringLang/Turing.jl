using Turing

function f_cta()
  t = TArray(Int, 1);
  t[1] = 0;
  while true
    produce(t[1])
    t[1]
    t[1] = 1 + t[1]
  end
end

t = Task(f_cta)

consume(t); consume(t)
a = copy(t);
consume(a); consume(a)

Base.@assert consume(t) == 2
Base.@assert consume(a) == 4

# Base.@assert TArray(Float64,  5)[1] != 0
Base.@assert tzeros(Float64, 5)[1]==0
Base.@assert tzeros(4)[1]==0
