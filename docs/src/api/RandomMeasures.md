# API: random measures

Access the extension after loading Turing:

```julia
using Turing

const RandomMeasures = Base.get_extension(Turing, :TuringDistributionsExt)
RandomMeasures.DirichletProcess(1.0)
```

```@autodocs
Modules = [Base.get_extension(Turing, :TuringDistributionsExt)]
Order  = [:type, :function]
```
