# NB: This requires mlr 2.5, which will hit CRAN soon
library(devtools)
load_all("~/cos/ParamHelpers")
load_all("~/cos/mlr")

# viper chars
lrn1 = makeLearner("classif.logreg", predict.type = "prob")
lrn2 = makeLearner("classif.rpart", predict.type = "prob")
b = benchmark(list(lrn1, lrn2), pid.task)
z = plotViperCharts(b, chart = "lift", browse = TRUE)


## feature importance example using breast cancer data
imp = generateFilterValuesData(bc.task, method = c("cforest.importance", "rf.importance"))
print(imp)
plotFilterValues(imp, n.show = 5)
plotFilterValuesGGVIS(imp)

## partial prediction example
n = getTaskSize(bc.task)
train_idx = seq(1, n, by = 2)
test_idx = seq(2, n, by = 2)
lrn = makeLearner("classif.randomForest", predict.type = "prob")
fit = train(lrn, bc.task, subset = train_idx)
bc = getTaskData(bc.task, subset = train_idx)
pd = generatePartialPredictionData(fit, data = bc, features = "Bare.nuclei")
pd_int = generatePartialPredictionData(fit, data = bc, features = c("Bare.nuclei", "Cell.size"))

plotPartialPrediction(pd)
plotPartialPrediction(pd_int, facet = "Cell.size")

## thresh versus perf example
pred = predict(fit, task = bc.task, subset = test_idx)
perf = generateThreshVsPerfData(pred, list(tpr, fpr, tnr, fnr))
plotThreshVsPerf(perf)
plotThreshVsPerfGGVIS(perf)

# learrning curve
r = generateLearningCurveData(list("classif.rpart", "classif.knn"),
 task = sonar.task, percs = seq(0.2, 1, by = 0.2),
 measures = list(tp, fp, tn, fn), resampling = makeResampleDesc(method = "Subsample", iters = 5),
 show.info = FALSE)
plotLearningCurve(r)
plotLearningCurveGGVIS(r)
