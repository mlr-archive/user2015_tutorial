### Basic hyperparameter comparison / tuning

library(mlr)

# we will compare values of k know for the data set
lrn = makeLearner("classif.svm")

# cross-validation, no need to pregenerate now.
# as this is sensible, tuneParams will do this for us automatically
rdesc = makeResampleDesc("CV", iters = 3L)

# Description of our parameter space we want to grid-search over
par.set = makeParamSet(
  makeNumericParam("cost",  lower = -15, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -15, upper = 15, trafo = function(x) 2^x)
)

# run it
# (actually, mlr supports many other tuner, so we need to select the tuner via a control object here)
ctrl = makeTuneControlGrid(resolution = 5L)
res = tuneParams(lrn, sonar.task, rdesc, par.set = par.set, control = ctrl)

# access full info of result
print(res)
print(names(res))
print(head(as.data.frame(res$opt.path)))



