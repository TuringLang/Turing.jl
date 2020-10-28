import .GalacticOptim

function instantiate_galacticoptim_function(model::DynamicPPL.Model, ::MAP , ::unconstrained)
    obj, init, t = instantiate_optimisation_problem(model, MAP() , unconstrained())
    
    l(x,p) = obj(x)
    f = GalacticOptim.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  
    return (f=f, init=init, transform = t)
  end
  
  function instantiate_galacticoptim_function(model::DynamicPPL.Model, ::MLE , ::unconstrained)
    obj, init, t = instantiate_optimisation_problem(model, MLE() , unconstrained())
    
    l(x,p) = obj(x)
    f = GalacticOptim.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  
    return (f=f, init=init, transform = t)
  end
  
  function instantiate_galacticoptim_function(model::DynamicPPL.Model, ::MAP , ::constrained)
    obj, init, t = instantiate_optimisation_problem(model, MAP(), constrained())
    
    l(x,p) = obj(x)
    f = GalacticOptim.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  
    return (f=f, init=init, transform = t)
  end
  
  function instantiate_galacticoptim_function(model::DynamicPPL.Model, ::MLE , ::constrained)
    obj, init, t = instantiate_optimisation_problem(model, MLE() , constrained())
    
    l(x,p) = obj(x)
    f = GalacticOptim.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  
    return (f=f, init=init, transform = t)
  end