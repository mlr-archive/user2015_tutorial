### tasks, training, predictions, evals

library(methods)
library(mlbench)
library(mlr)

### lets use the famous iris data set
task = makeClassifTask(data = iris, target = "Species")
print(task)

#### lets use lda as a learner

lrn = makeLearner("classif.lda")
print(getParamSet(lrn))

### train + predict + eval

# train
mod = train(lrn, task, subset = seq(1, 150, 2))
print(mod)
# access real lda model
print(mod$learner.model)

# predict
pred = predict(mod, task, subset = seq(2, 150, 2))
print(str(as.data.frame(pred)))

# eval
p = performance(pred, measures = mmce)
print(p)
print(getConfMatrix(pred))

# some basic EDA and preprocessing

data("Ionosphere", package = "mlbench")
task = makeClassifTask(data = Ionosphere, target = "Class", positive = "good")
print(task)
summarizeColumns(task)
task2 = removeConstantFeatures(task, show.info = TRUE)
?summarizeColumns # look at other methods in family
