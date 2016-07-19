using Turing, DataFrames, Gadfly, Distributions

# HMM example from Anglican and Probabilistic C papers

statesmean = [-1, 1, 0]
initial    = Categorical([1.0/3, 1.0/3, 1.0/3])
trans      = [Categorical([0.1, 0.5, 0.4]), Categorical([0.2, 0.2, 0.6]), Categorical([0.15, 0.15, 0.7])]
data       = [0, 0.9, 0.8, 0.7, 0, -0.025, -5, -2, -0.1, 0, 0.13]

@model hmmdemo begin
  states = TArray(Int, length(data))
  @assume states[1] ~ initial
  for i = 2:length(data)
    @assume states[i] ~ trans[states[i-1]]
    @observe data[i]  ~ Normal(statesmean[states[i]], 0.4)
  end
  @predict states data
end

#  run sampler, collect results
@time chain1  = sample(hmmdemo, SMC(500))
@time chain2  = sample(hmmdemo, PG(10, 20))

SMCInferedStates  = mean([statesmean[v] for v in chain1[:states]])
PGInferedStates   = mean([statesmean[v] for v in chain2[:states]])
println("Infered states  (smc)", round(SMCInferedStates, 2))
println("Infered states   (pg)", round(PGInferedStates, 2))
println("True states         ", round(data+0.01, 2))



@model gaussdemo begin
  # Define a simple Normal model with unknown mean and variance.
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

@time chain1 = sample(gaussdemo, IS(200))

program = expand(Turing.TURING[:modelex])
model_f = eval(program)

model_f()
gaussdemo()

:($(Expr(:method, :gaussdemo, :((top(svec))((top(apply_type))(Tuple),(top(svec))())), AST(:($(Expr(:lambda, Any[], Any[Any[Any[:s,:Any,18],Any[:m,:Any,18],Any[:ct,:Any,2]],Any[],0,Any[]], :(
begin
  s = (Turing.assume)(Turing.sampler,(Main.InverseGamma)(2,3))
  m = (Turing.assume)(Turing.sampler,(Main.Normal)(0,(Main.sqrt)(s)))
  (Turing.observe)(Turing.sampler,(Main.logpdf)((Main.Normal)(m,(Main.sqrt)(s)),1.5))
  (Turing.observe)(Turing.sampler,(Main.logpdf)((Main.Normal)(m,(Main.sqrt)(s)),2.0))
  ct = (Main.current_task)()
  (Turing.predict)(Turing.sampler,(Main.symbol)("s"),(Main.get)(ct,s))
  ct = (Main.current_task)()
  (Turing.predict)(Turing.sampler,(Main.symbol)("m"),(Main.get)(ct,m))
  return (Main.produce)((top(apply_type))(Main.Val,:done))
end))))), false)))


ex = Expr(:method, :aa, 1, 2)
eval(ex)

using Turing, Distributions
@model model_exp begin
  @assume s ~ Normal(0, 1)
end


expand(Turing.TURING[:modelex])

a = []
ct = current_task()
get(ct, a)

Turing.sampler = IS(200)
@assume s ~ Normal(0, 1)



function producer()
  produce("start")
  for n=1:4
    produce(2n)
  end
  produce("stop")
end;

p = Task(producer)
consume(p)

ct = current_task()
istaskdone(ct)
istaskstarted(ct)
task_local_storage()



using Turing, Distributions, DualNumbers

xs = rand(Normal(0.5, 4), 500)
@model gausstest begin
  @assume s ~ InverseGamma(2, 3)
  @assume m ~ Normal(0, sqrt(s))
  for x in xs
    @observe x ~ Normal(m, sqrt(s))
  end
  @predict s m
end

# HMC(n_samples, lf_size, lf_num)
chain = sample(gausstest, HMC(1000, 0.01, 5))
s = mean([d[:s] for d in chain[:samples]])
m = mean([d[:m] for d in chain[:samples]])

chain2 = sample(gausstest, SMC(100))
mean([d[:s] for d in chain2[:samples]])
mean([d[:m] for d in chain2[:samples]])


f = Normal(0, sqrt(1))
show(f)

modelex = Turing.TURING[:modelex]
print(modelex)
# >
# function gausstest()
#     @assume @~(s,InverseGamma(2,3))
#     @observe @~(1.5,Normal(0,sqrt(s)))
#     @observe @~(2.0,Normal(0,sqrt(s)))
#     @predict s
#     produce(Val{:done})
# end

# This step is done by marco model
modelex_expanded = expand(modelex)
print(modelex_expanded)

# AST: Abstract Syntax Tree

# >
# $(
#
# Expr(
#   :method,
#   :gausstest,
  # :(  (top(svec))( (top(apply_type))(Tuple), (top(svec))() )  ),
#   AST(
#     :(
#       $(
#         Expr(
#         :lambda,
#         Any[],
#         Any[Any[Any[:s, :Any, 18],
#         Any[:ct, :Any, 18]],
#         Any[],0,
#         Any[]],
#         :(begin
#             s = (Turing.assume)(
#               Turing.sampler,
#               (Main.InverseGamma)(2, 3)
#             )
#             (Turing.observe)(
#               Turing.sampler,
#               (Main.logpdf)((Main.Normal)(0, (Main.sqrt)(s)), 1.5)
#             )
#             (Turing.observe)(
#               Turing.sampler,
#               (Main.logpdf)((Main.Normal)(0, (Main.sqrt)(s)), 2.0)
#             )
#             ct = (Main.current_task)()
#             (Turing.predict)(
#               Turing.sampler, (Main.symbol)("s"), (Main.get)(ct, s)
#             )
#             return (Main.produce)((top(apply_type))(Main.Val, :done))
#           end)
#         )
#       )
#     )
#   ),
#   false)
#
# )
typeof(modelex_expanded)
tt = eval(modelex_expanded)
typeof(tt)
show(tt)

chain = sample(tt, IS(5))
Turing.sampler

tt()

s = (Turing.assume)(
              Turing.sampler,
              (Main.InverseGamma)(2, 3)
    )

(Turing.observe)(
  Turing.sampler,
  (Main.logpdf)((Main.Normal)(0, (Main.sqrt)(s)), 1.5)
)

produce(1.0)
