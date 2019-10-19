library(zoo)
library(lattice)
library(rpart)
library(ggplot2)
library(caret)
library(rattle)					# Fancy tree plot
library(rpart.plot)			# Enhanced tree plots
library(RColorBrewer)		# Color selection for fancy tree plot
library(party)					# Alternative decision tree algorithm
library(AUC)
library(pROC)
library(plotROC)

#Reading from Database
dados = read.csv("/home/rosana/Documents/Mestrado/Projeto Disciplina/Base-Final-Rotulada.csv",sep=",", header=T)
head(dados)

#Sum Class
summary(factor(dados$classes))

## 70% of the sample size
sample_size <- floor(0.7 * nrow(dados))

## set the seed to make your partition reproductible
set.seed(123422)
train_index <- sample(seq_len(nrow(dados)), size = sample_size)

x_train <- dados[train_index, ]
x_test <- dados[-train_index, ]

y_train <- x_train$classes
y_test  <- x_test$classes

x_train$classes<-NULL
x_test$classes<-NULL

# Create a stratified sample for repeated cv
#times: the number of partitions to create
#k: an integer for the number of folds.
#y_train: a vector of outcomes.
cv_10_folds<-createMultiFolds(y_train,k=10,times=2)

# create a control object for repeated cv in caret
ctrl <- trainControl(method="repeatedcv",number=10,repeats=3,index=cv_10_folds)

tree <-train(x=x_train,y=y_train,method="rpart",trControl=ctrl,tuneLength=5, cp=0.1)

y_predicted_tree<-predict(tree,x_test)
df<-data.frame(Orig=y_test,Pred=y_predicted_tree)
confusionMatrix(table(df$Orig,df$Pred))
plot(tree)
fancyRpartPlot(tree$finalModel, main="Árvore de decisão")


#    RANDOM FOREST
rf <-train(x=x_train,y=y_train,method="rf",trControl=ctrl)

y_predicted_rf<-predict(rf,x_test)
df<-data.frame(Orig=y_test,Pred=y_predicted_rf)
confusionMatrix(table(df$Orig,df$Pred))
plot(rf)
# Variáveis mais significantes para o modelo
VarImportance <- varImp(rf)
# Plot the top 15 predictors
plot(VarImportance, main = "Variáveis mais significantes para o modelo Random Forest", top = 7)


#     BOOSTED TREE
boosted <-train(x=x_train,y=y_train,method="gbm",trControl=ctrl,tuneLength=5)

y_predicted_bt<-predict(boosted,x_test)
df<-data.frame(Orig=y_test,Pred=y_predicted_bt)
confusionMatrix(table(df$Orig,df$Pred))
plot(boosted, main="Boosted")

# Acurácia ##################
Model <- c("CART", "Random Forest","Boosted Tree")
Accuracy <- c(round(max(tree$results$Accuracy),4)*100,round(max(rf$results$Accuracy),4)*100,
              round(max(boosted$results$Accuracy),4)*100 )
Performance <- cbind(Model,Accuracy); 
Performance

#ROC curve - AUC
