### Complex hyperparameter tuning with wrapper and nested resampling

library(mlr)
library(mlbench)


# param set for an svm, multiple kernel, dependent parameters and trafos
lrn = makeLearner("classif.svm")
ps = makeParamSet(
  makeDiscreteParam("kernel", values = c("polynomial", "radial")),
  makeNumericParam("cost", lower = -15, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -15, upper = 15, trafo = function(x) 2^x,
   requires = expression(kernel == "radial")),
  makeIntegerParam("degree", lower = 1, upper = 5,
   requires = expression(kernel == "polynomial"))
)


# we can use irace or a simple random search here
# ctrl = makeTuneControlIrace(maxExperiments = 100L)
ctrl = makeTuneControlRandom(maxit = 20)

# this adds the tuning to the learner, we use holdout on inner resampling
inner = makeResampleDesc(method = "Holdout")
lrn2 = makeTuneWrapper(lrn, inner, par.set = ps, control = ctrl, measures = mmce)

# now run everything, we use CV with 2 folds on the outer loop
outer = makeResampleDesc(method = "CV", iters = 2)
r = resample(lrn2, sonar.task, outer, extract = getTuneResult)
print(r)

# lets look at some results from the outer iterations
r$extract[[1]]$x
r$extract[[1]]$y
r$extract[[1]]$opt.path


