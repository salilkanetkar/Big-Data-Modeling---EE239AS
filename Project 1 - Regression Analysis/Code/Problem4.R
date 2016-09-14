#Problem4
################################################################################
library('usdm')
h_pre_model <- lm(MEDV~.,data=housing_data)
summary(h_pre_model)
h_main_model <- lm(MEDV ~ ZN+NOX+RM+DIS+RAD+PTRATIO+B+LSTAT,data=housing_data)
summary(h_main_model)
h_final_model <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,data=housing_data)
summary(h_final_model)
#vif(h_final_model)
rmse_ols <- sqrt(sum((housing_data$MEDV-h_final_model$fitted.values)^2)/nrow(housing_data))
################################################################################

#To cross validate, we sampled the data and randomized it
randomized_data<-housing_data[sample(nrow(housing_data)),]
#As mentioned in the question, created 10 folds
folds <- cut(seq(1,nrow(randomized_data)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  test_indices <- which(folds==i,arr.ind=TRUE)
  testing_dataset <- randomized_data[test_indices, ]
  training_dataset <- randomized_data[-test_indices, ]
  cross_validation_model <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,training_dataset) #Train the Model on 90% of the data
  cv_pred <- predict(cross_validation_model,testing_dataset) #Test on the% of the data
  row_names <- rownames(as.matrix(cv_pred))
  for(j in 1:length(cv_pred))
  {
    fitted_values[as.numeric(row_names[j])] = cv_pred[j] #Store the fitted values in an array
  }
}
actual_values <- housing_data$MEDV
rmse_cv <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data)) #Calculate the RMSE
plot(cross_validation_model)

################################################################################

#CV Using in-built function
library('DAAG')
cv_model<-cv.lm(data = housing_data, form.lm = formula
                (MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT),
                m = 10)
predicted_values <- cv_model$cvpred
actual_values <- housing_data$MEDV
cv_model_inbuilt_rmse <- sqrt(sum((actual_values-predicted_values)^2)/nrow(housing_data))

################################################################################

library(lattice)
res <- stack(data.frame(Observed = housing_data$MEDV, Predicted = predicted_values))
x = data.frame(cbind(1:506,housing_data$NOX,housing_data$RM,housing_data$DIS,housing_data$PTRATIO,housing_data$B,housing_data$LSTAT))
res <- cbind(res, x = rep(x$X1, 2))
res <- cbind(res, x = rep(x$X2, 2))
res <- cbind(res, x = rep(x$X3, 2))
res <- cbind(res, x = rep(x$X4, 2))
res <- cbind(res, x = rep(x$X5, 2))
res <- cbind(res, x = rep(x$X6, 2))
#x <- data.frame(1:506)
y <- housing_data$MEDV
xyplot(values ~ x, data = res, group = ind, auto.key = TRUE,xlab='Index',ylab='MEDV')
xyplot(y ~ x, data = res, type = c("p","r"), col.line = "red")

################################################################################

#Polynomial Regression
ds <- housing_data
randomized_data<-ds[sample(nrow(ds)),] #Randomly distributing the data
poly_rmse_all = NULL
#Create 10 equally size folds
folds <- cut(seq(1,nrow(randomized_data)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(ds),ncol=1)
#Perform 10 fold cross validation
for(a in 1:10)
{
  fitted_values = matrix(data=NA,nrow=nrow(ds),ncol=1)
  rmse = NULL
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    test_indices <- which(folds==i,arr.ind=TRUE)
    training_dataset <- (randomized_data[-test_indices, ])
    cv_poly_model <- lm(MEDV ~ polym(NOX,RM,DIS,PTRATIO,B,LSTAT,degree = a,raw = TRUE),data=training_dataset)
    training_dataset <- randomized_data[test_indices, ]
    cv_pred <- predict(cv_poly_model,training_dataset)
    name <- rownames(as.matrix(cv_pred))
    for(j in 1:length(cv_pred))
    {
      fitted_values[as.numeric(name[j])] = cv_pred[j] #String the fitted values
    }
  }
  actual_values <- ds$MEDV
  rmse <- sqrt(mean((actual_values-fitted_values)^2))
  poly_rmse_all <- append(poly_rmse_all,rmse)
}

################################################################################