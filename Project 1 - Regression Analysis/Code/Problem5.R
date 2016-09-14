#Problem5
################################################################################
library("MASS")
lam = seq(0,10000,len=5000)
ridgefits = lm.ridge(housing_data$MEDV~.,data=housing_data,lam=lam)
plot(range(lam), range(ridgefits$coef),type="n",xlab='Lambda', ylab='Coefficients')
for(i in 1:nrow(ridgefits$coef))
  lines(lam,ridgefits$coef[i,])

################################################################################
library(glmnet)
#To cross validate, we sampled the data and randomized it
randomized_data<-housing_data[sample(nrow(housing_data)),]
#As mentioned in the question, created 10 folds
folds <- cut(seq(1,nrow(randomized_data)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  test_indices <- which(folds==i,arr.ind=TRUE)
  training_dataset <- randomized_data[-test_indices, ]
  x = as.matrix(training_dataset[, (colnames(training_dataset) %in% c("NOX","RM","DIS","PTRATIO","B","LSTAT"))])
  #cross_validation_model <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,training_dataset) #Train the Model on 90% of the data
  cross_validation_model <- glmnet(x, training_dataset$MEDV, family="gaussian", alpha=0.01, lambda=0.001)
  training_dataset <- randomized_data[test_indices, ]
  x = as.matrix(training_dataset[, (colnames(training_dataset) %in% c("NOX","RM","DIS","PTRATIO","B","LSTAT"))])
  #cv_pred <- predict(cross_validation_model,training_dataset) #Test on the% of the data
  cv_pred <- predict(cross_validation_model, x, type="link")
  row_names <- rownames(as.matrix(cv_pred))
  for(j in 1:length(cv_pred))
  {
    fitted_values[as.numeric(row_names[j])] = cv_pred[j] #Store the fitted values in an array
  }
}
actual_values <- housing_data$MEDV
rmse_cv <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data)) #Calculate the RMSE

################################################################################


#LASSO_Regression
library("lars")
#To cross validate, we sampled the data and randomized it
randomized_data<-housing_data[sample(nrow(housing_data)),]
#As mentioned in the question, created 10 folds
folds <- cut(seq(1,nrow(randomized_data)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  test_indices <- which(folds==i,arr.ind=TRUE)
  training_dataset <- randomized_data[-test_indices, ]
  x = as.matrix(training_dataset[, (colnames(training_dataset) %in% c("NOX","RM","DIS","PTRATIO","B","LSTAT"))])
  #cross_validation_model <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,training_dataset) #Train the Model on 90% of the data
  cross_validation_model <- lars(x, training_dataset$MEDV, type="lasso")
  best_step <- cross_validation_model$df[which.min(cross_validation_model$RSS)]
  training_dataset <- randomized_data[test_indices, ]
  x = as.matrix(training_dataset[, (colnames(training_dataset) %in% c("NOX","RM","DIS","PTRATIO","B","LSTAT"))])
  #cv_pred <- predict(cross_validation_model,training_dataset) #Test on the% of the data
  cv_pred <- predict(cross_validation_model, x, s=best_step, type="fit")$fit
  row_names <- rownames(as.matrix(cv_pred))
  for(j in 1:length(cv_pred))
  {
    fitted_values[as.numeric(row_names[j])] = cv_pred[j] #Store the fitted values in an array
  }
}
actual_values <- housing_data$MEDV
rmse_cv <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data)) #Calculate the RMSE
################################################################################