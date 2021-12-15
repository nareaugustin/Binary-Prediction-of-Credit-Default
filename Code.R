rm(list = ls())

library(ggplot2)
library(caret)
library(glmnet)
library(rpart)
library(tidyverse)
library(randomForest)
library(dplyr)
library(gridExtra)
library(grid)
library(modelr)
library(reshape)
library(ggthemes)
library(pROC)

###Imports data
data <- read.csv("data.csv")


###Gets rid of NULL columns and unecessary columns

data = subset(data, select = -c(initial_list_status,issue_d,earliest_cr_line ,last_pymnt_d , next_pymnt_d, title ,emp_title, member_id, id,desc, mths_since_last_delinq,mths_since_last_record, mths_since_last_major_derog,annual_inc_joint,dti_joint,tot_coll_amt, tot_cur_bal,open_acc_6m, open_il_6m, open_il_12m , open_il_24m , mths_since_rcnt_il ,total_bal_il ,il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util, total_rev_hi_lim, inq_fi, total_cu_tl, inq_last_12m,verification_status_joint,last_credit_pull_d, zip_code) )

# delete empty rows
data = na.omit(data)


#For test purposes, lower amount of data
data <- data[1:1000,]

# transformation of categorical variable

data$pymnt_plan[data$pymnt_plan == "n"] <- 0
data$pymnt_plan[data$pymnt_plan == "y"] <- 1
data$pymnt_plan = factor(data$pymnt_plan)

data$grade = factor(data$grade)

data$term[data$term == " 36 months"] <- 0
data$term[data$term == " 60 months" ] <- 1
data$term = factor(data$term)

data$sub_grade = factor(data$sub_grade)
data$verification_status = factor(data$verification_status)
data$addr_state = factor(data$addr_state)
data$application_type[data$application_type == "INDIVIDUAL"] <- 0
data$application_type[data$application_type == " JOINT"] <- 1
data$application_type= factor(data$application_type)
data$purpose = factor(data$purpose)
data$emp_length = factor(data$emp_length)
data$home_ownership = factor(data$home_ownership)

###Setting to numeric values only
for (i in 1:ncol(data)) {
  if (is.factor(data[,i])){
    data[,i] <- as.numeric(data[,i])
  }
  
}

#Split up the X variables and the Y variable into a matrix and a vector, respectively. 
#Using the model.matrix() function, creates dummy variables out of the categorical variables.


x = model.matrix(default_ind~., data)[,-38]
y = data$default_ind

#Identify n and p. Identify the training and test sizes. (90/10)

n = nrow(x)    # number of total observations
p = ncol(x)    # number of predictors
n.train = floor(0.9 * n)
n.test  = n - n.train


S = 50 #iterations of loop
#for testing

#LASSO AUC & TIME TAKEN
AUC.test.la     <- rep(0, S)  
AUC.train.la    <- rep(0, S)

AUC.test.en     <- rep(0, S)
AUC.train.en    <- rep(0, S)

AUC.test.rid     <- rep(0, S)
AUC.train.rid    <- rep(0, S)

#M = # of times to run loop  
M              =     1        
AUC.test.rf     <- rep(0, S)
AUC.train.rf    <- rep(0, S)


#############Lasso Regression##########



for (s in c(1:S)) {
  
  cat("s = ", s, "\n")
  
  shuffled_indexes <-     sample(n)
  train            <-     shuffled_indexes[1:n.train]
  test             <-     shuffled_indexes[(1+n.train):n]
  X.train          <-     x[train, ]
  y.train          <-     y[train]
  X.test           <-     x[test, ]
  y.test           <-     y[test]
  
  n.P                 =        sum(y.train)
  n.N                 =        n.train - n.P
  ww                  =        rep(1,n.train)
  ww[y.train==1]      =        n.N/n.P
  
  ###################################################################
  ##### Fit lasso and calculate and record the train and test AUC,
  #############
  
  cv.lasso    =      cv.glmnet(X.train, y.train, alpha = 1, family = "binomial", intercept = TRUE,  nfolds = 10,weights = ww,trace.it=1)
  
  lasso.fit   =      glmnet(X.train, y.train, lambda = cv.lasso$lambda.min, family = "binomial", alpha = 1,  intercept = TRUE, weights = ww,trace.it=1)
  
  ###Predict train data based on model
  lasso_train_probs <- predict(lasso.fit, X.train, type = "response")
  
  ###roc for train
  roc_Lasso_train <- roc(y.train, lasso_train_probs)
  
  ###Stores AUC
  AUC.train.la[s] <- roc_Lasso_train$auc
  
  ###Predict test data based on model
  lasso_test_probs <- predict(lasso.fit, X.test, type = "response")
  
  ###Roc for test
  roc_Lasso_test <- roc(y.test, lasso_test_probs)
  
  ###Test AUC
  AUC.test.la[s] <- roc_Lasso_test$auc
  
  ###################################################################
  ##### Fit EN and calculate and record the train and test AUC,
  #############
  
  cv.en    =      cv.glmnet(X.train, y.train, alpha = 0.5, family = "binomial", intercept = TRUE,  nfolds = 10,weights = ww,trace.it=1)
  
  en.fit   =      glmnet(X.train, y.train, lambda = cv.en$lambda.min, family = "binomial", alpha = 0.5,  intercept = TRUE, weights = ww,trace.it=1)
  
  ###Predict train data based on model
  en_train_probs <- predict(en.fit, X.train, type = "response")
  
  ###roc for train
  roc_en_train <- roc(y.train, en_train_probs)
  
  ###Stores AUC
  AUC.train.en[s] <- roc_en_train$auc
  
  ###Predict test data based on model
  en_test_probs <- predict(en.fit, X.test, type = "response")
  
  ###Roc for test
  roc_en_test <- roc(y.test, en_test_probs)
  
  ###Test AUC
  AUC.test.en[s] <- roc_en_test$auc
  
  ###################################################################
  ##### Fit ridge and calculate and record the train and test AUC,
  #############
  
  cv.rid    =      cv.glmnet(X.train, y.train, alpha = 0, family = "binomial", intercept = TRUE,  nfolds = 10,weights = ww,trace.it=1)
  
  rid.fit   =      glmnet(X.train, y.train, lambda = cv.rid$lambda.min, family = "binomial", alpha = 0.5,  intercept = TRUE, weights = ww,trace.it=1)
  
  ###Predict train data based on model
  rid_train_probs <- predict(rid.fit, X.train, type = "response")
  
  ###roc for train
  roc_rid_train <- roc(y.train, rid_train_probs)
  
  ###Stores AUC
  AUC.train.rid[s] <- roc_rid_train$auc
  
  ###Predict test data based on model
  rid_test_probs <- predict(en.fit, X.test, type = "response")
  
  ###Roc for test
  roc_en_test <- roc(y.test, rid_test_probs)
  
  ###Test AUC
  AUC.test.rid[s] <- roc_en_test$auc
  
  
  ###################################################################
  ##### Fit RF and calculate and record the train and test AUC,
  #############
  rf_data_train <- data[train,]
  rf_data_train$default_ind <- as.factor(rf_data_train$default_ind)
  
  rf_data_test <- data[test,]
  rf_data_test$default_ind <- as.factor(rf_data_test$default_ind)
  
  p = ncol(rf_data_train)
  rf <- randomForest(default_ind ~ . ,data = rf_data_train, mtry = sqrt(p)) 
  
  ####ROC AUC Train
  roc_train <- roc(rf_data_train$default_ind, rf$votes[,2])
  AUC.train.rf[s] <- roc_train$auc
  
  ###ROC AUC Test
  rf_predict_test <- predict(rf, rf_data_test, type = 'prob') 
  rf_predict_test <- data.frame(rf_predict_test)
  
  rf_roc_test <- roc(rf_data_test$default_ind, rf_predict_test$X0)
  AUC.test.rf[s] <- rf_roc_test$auc
  
}


# Fit lasso and record CV time
start.time       <-     Sys.time()
cv.lasso         <-     cv.glmnet(X.train , y.train,family = "binomial", alpha=1, intercept = TRUE, nfolds = 10, weights = ww)
lasso.fit        <-     glmnet(X.train,y.train,alpha=0.5, family = "binomial",lambda=cv.lasso$lambda, weights = ww)
end.time         <-     Sys.time()
time.lasso       <-     end.time - start.time

# Fit elastic-net and record CV time
start.time       <-     Sys.time()
cv.en            <-     cv.glmnet(X.train , y.train,family = "binomial", alpha=0.5, intercept = TRUE, nfolds = 10, weights = ww)
elnet.fit        <-     glmnet(X.train,y.train,alpha=0.5, family = "binomial",lambda=cv.en$lambda, weights = ww)
end.time         <-     Sys.time()
time.en          <-     end.time - start.time


#Fit ridge and calculate and record the train and test AUC, 
start.time       <-     Sys.time()
cv.ridge         =      cv.glmnet(X.train, y.train, family = "binomial", alpha = 0,  intercept = TRUE,   nfolds = 10, weights = ww)
ridge.fit        =      glmnet(X.train,y.train,alpha=0,family = "binomial",  lambda=cv.ridge$lambda, weights = ww)
end.time         <-     Sys.time()
time.ridge       <-     end.time - start.time

# Fit RF and calculate and record the train and test AUC,  
start.time       <-     Sys.time()
rf               <-     randomForest(default_ind ~ . ,data = rf_data_train, mtry = sqrt(p))
end.time         <-     Sys.time()
time.rf          <-     end.time - start.time

time.ridge
time.en 
time.lasso 

par(mfrow=c(1,3))
plot(cv.fit, main = "Elastic Net")
plot(cv.ridge , main = "Ridge")
plot(cv.lasso, main = "lasso")



#########################
####CV Plot 
#########################
par(mfcol = c(1, 3))
plot(cv.lasso)
title("Lasso", line = 3)

plot(cv.ridge)
title("Ridge", line = 3)

plot(cv.en)
title("Elastic net", line = 3)

#######################
####AUC BoxPlots
#######################
library(ggplot2)
library(reshape)

training_auc   =  data.frame(AUC.train.en, AUC.train.la, AUC.train.rid, AUC.train.rf)
test_auc       =  data.frame(AUC.test.en, AUC.test.la, AUC.test.rid, AUC.test.rf)

mdata_train <- melt(training_auc)
mdata_test <- melt(test_auc)

#Create box plot of the training AUC values of the different models 
plot_train = ggplot(mdata_train, aes(x = variable, y= value)) +
  geom_boxplot() + 
  labs(title = 'Training Data AUC', y = 'AUC', x = 'Model') +
  scale_x_discrete(labels = c("Elastic Net","Lasso","Ridge","Random Forest"))
plot_train 

#Create box plot of the test AUC values of the different models   
plot_test = ggplot(mdata_test, aes(x = variable, y= value)) +
  geom_boxplot() + 
  labs(title = 'Testing Data AUC', y = 'AUC', x = 'Model')+
scale_x_discrete(labels = c("Elastic Net","Lasso","Ridge","Random Forest"))
plot_test 

Coef.Plot=grid.arrange(plot_test, plot_train, ncol = 2)



###5.3##
#Bar plots of the estimated coefficients, importance of the parameters
#obtaining estimated coefficients for each model
p = ncol(X.train)
n = nrow(X.train) 
#Coefficients Ridge
beta.coef_rid=data.frame(c(1:p), as.vector(ridge.fit$beta))
colnames(beta.coef_rid)=c("feature", "value")

#Coefficients Lasso
beta.coef_las=data.frame(c(1:p), as.vector(lasso.fit$beta))
colnames(beta.coef_las)=c("feature", "value")

#Coefficients El-net
beta.coef_elnet=data.frame(c(1:p), as.vector(en.fit$beta))
colnames(beta.coef_elnet)=c("feature", "value")

#Coefficients Random Forest
beta.coef_rf=data.frame(c(1:p), as.vector(rf$importance[1:p]))
colnames(beta.coef_rf)=c("feature", "value")

#changing the order of factor levels by specifying the order explicitly
# specifically use elastic-net estimated coefficients to create an order based on descending order
# use this order to present plots of the estimated coefficients of all 4 models
# Ridge

beta.coef_rid$feature=factor(beta.coef_rid$feature, levels = beta.coef_elnet$feature[order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_elnet$feature=factor(beta.coef_elnet$feature, levels = beta.coef_elnet$feature[order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_las$feature=factor(beta.coef_las$feature, levels = beta.coef_elnet$feature[order(beta.coef_elnet$value, decreasing = TRUE)])
beta.coef_rf$feature=factor(beta.coef_rf$feature,levels = beta.coef_elnet$feature[order(beta.coef_elnet$value, decreasing = TRUE)])

#Coefficients plots

# Elastic-Net Plot
ElNetPlot=ggplot(beta.coef_elnet, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="yellow", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic-Net))+
  ylim(-2,6)

RidPlot=ggplot(beta.coef_rid, aes(x=feature, y=value)) +
  geom_bar()+
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge))+
  ylim(-2,6)

# Lasso Plot
LasPlot=ggplot(beta.coef_las, aes(x=feature, y=value)) +
  geom_bar()+
  labs(x = element_blank(), y = "Coefficients", title = expression(Lasso))+
  ylim(-2,6)


# Random Forrest
RFPlot=ggplot(beta.coef_rf, aes(x=feature, y=value)) +
  geom_bar()    +
  labs(x = element_blank(), y = "Importance", title = expression(RandomForest))

Coef.Plot=grid.arrange(RidPlot, LasPlot, ElNetPlot, RFPlot, nrow = 4)
importance(rf)


# Median AUCs for testing 

median(AUC.test.en)
median(AUC.test.la)
median(AUC.test.rid)
median(AUC.test.rf)
