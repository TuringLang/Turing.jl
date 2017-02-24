using Turing

include("config.jl")

if CONFIG["test-level"] == 1

  println("Model language tests started.")

  for model in CONFIG["model-list"]
    print("$model is being tested ... ")
    include("$model/model.jl")
    println("✓")
  end

  println("Model language tests finished.")

elseif CONFIG["test-level"] == 2

  println("Inference interface tests started.")

  for model in CONFIG["model-list"]
    print("$model is being tested ... ")
    include("$model/run.jl")
    println("✓")
  end

  println("Inference interface tests finished.")

end
