rm(list = ls())
library(R6)
library(coro)
library(ggplot2)
library(boot)
library(dplyr)
library(pROC)
library(ROCR)

sigmoid = function(x){
  return(1 / (1 + exp(-x)))
}

logit = function(x, w){
  return(x %*% w)
}

sign = function(x){
  if(x>0)
    return(1)
  else if (x==0)
    return(0)
  else
    return(-1)
}

LogRegression = R6Class(classname = "LR",
                        public = list(
                          ###
                          w = NULL,
                          accus = c(),
                          losses = c(),
                          ###
                          setW = function(x){
                            self$w = rnorm(x)
                            invisible(self)
                          },
                          ###
                          getW = function(){
                            return(self$w)
                            invisible(self)
                          },
                          ###
                          get_grad = function(x, y, preds){
                            x = as.matrix(x)
                            y = as.matrix(y)
                            preds = as.matrix(preds)
                            return(t(x) %*% (preds - y)) #+ 2 * self$w * 0.4 + 0.4 * sign(self$w))
                          },
                          ###
                          fit = function(x, y, lr = 0.1, epochs = 200, batch.size = 50, x.test, y.test){
                            vec = rep(1, nrow(x))
                            numCols = ncol(x)
                            x = data.frame(c(x, "bias" = vec))[0:numCols + 1]
                            for (ep in 1:epochs){
                              for (i in seq(1, nrow(x) - batch.size, batch.size)){
                                temp.n = i
                                temp.k = i + batch.size
                                x.batch = as.matrix(x[temp.n : temp.k, ])
                                y.batch = as.vector(y[temp.n : temp.k])
                                preds = sigmoid(logit(x.batch, self$w))
                                preds = as.matrix(preds)
                                #loss = -(y.batch %*% log(preds) - (1 - y.batch) %*% log(1 - preds))
                                #self$losses = c(self$losses, loss) 
                                self$w = self$w - self$get_grad(x.batch, y.batch, preds) * lr
                                
                                
                                acc = mean(round(preds) == y.batch)
                                self$accus = c(self$accus, acc)
                              }
                              
                            }
                          },
                          ###
                          predict = function(x){
                            vec = rep(1, nrow(x))
                            numCols = ncol(x)
                            x = data.frame(c(x, "bias" = vec))[0:numCols + 1]
                            x = as.matrix(x)
                            self$w = as.matrix(self$w)
                            return(round(sigmoid(logit(x, self$w))))
                          }
                        )
)

### Обучение
df = read.csv("/Users/test/RsProjects/kurs/train.csv", encoding = "UTF-8")

for (i in 1:nrow(df)){
  if (df$Sex[i] == "male"){
    df$Sex[i] = 1
  }
  else{
    df$Sex[i] = 0
  }
}

for (i in 1:nrow(df)){
  if (df$Embarked[i] == "C"){
    df$Embarked[i] = 1
  }
  else if (df$Embarked[i] == "S"){
    df$Embarked[i] = 2
  }
  else{
    df$Embarked[i] = 3
  }
}

df$"Sex" = as.numeric(df$"Sex")
df$"Embarked" = as.numeric(df$"Embarked")

df = select(df, c(Age, Sex, Pclass, Parch, SibSp, Fare, Embarked, Survived))
str(df)

df = df[is.finite(rowSums(df)),]

#df = data.frame(scale(df, center=TRUE, scale = FALSE))

ggplot(df, aes(x = Survived)) + geom_histogram(fill = "green", col = "black") +
  labs(title = "Распределение survived", 
       x = "Значение Survived", 
       y = "Количество наблюдений")

x.train = df[1:7][1:600, ]
y.train = df$Survived[1:600]
x.test = df[1:7][601:714, ]
y.test = df$Survived[601:714]

model = LogRegression$new()
model$setW(ncol(x.train) + 1)
model$getW()
model$fit(x.train, y.train, x.test = x.test, y.test = y.test, epochs = 2, lr=0.01, batch.size = 20)

mean(round(y.test) == model$predict(x.test))

dfroc = data.frame("true" = y.test, "pred" = model$predict(x.test))
pROC_obj <- roc(dfroc$true,dfroc$pred,
                smoothed = TRUE,
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


dfAUC = data.frame(x = seq(length(model$accus)), y = model$accus)
dfAUC = dfAUC[is.finite(rowSums(dfAUC)),]
ggplot(dfAUC, aes(x = x , y = y)) + geom_line(col = "red") +
  labs(title = "Accuracy во время обучения", 
       x = "Номер итерации/батча", 
       y = "Accuracy") + geom_hline(yintercept = mean(dfAUC$y), color = "green")
  

df.glm = df[1:600, ]
glm.fit = glm(Survived ~ Age + Sex + Pclass + Parch + SibSp + Fare + Embarked, data = df.glm, family = binomial)
glm.pred = predict.glm(glm.fit, newdata = x.test)
glm.pred = ifelse(glm.pred > 0.5, 1, 0)

pROC_obj <- roc(dfroc$true,glm.pred,
                smoothed = TRUE,
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=T, grid=TRUE,
                print.auc=TRUE, show.thres=T)

