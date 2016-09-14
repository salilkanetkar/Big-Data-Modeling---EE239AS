#Problem2A
#We tranied the model with all the parameters to observe the important ones
 ################################################################################
pre_model <- lm(Size.of.Backup..GB. ~ Week.. + Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID+File.Name+Backup.Time..hour.,network_backup_dataset)
summary(pre_model)

################################################################################
#The model below is trained with only the relevant parameters
main_model <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,network_backup_dataset)
summary(main_model)
rmse_main <- sqrt(sum((network_backup_dataset$Size.of.Backup..GB.-main_model$fitted.values)^2)/nrow(network_backup_dataset))
plot(main_model)
#We got an RMSE of 0.07404
################################################################################
#To cross validate, we sampled the data and randomized it
randomized_data<-network_backup_dataset[sample(nrow(network_backup_dataset)),]
#As mentioned in the question, created 10 folds
folds <- cut(seq(1,nrow(randomized_data)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(network_backup_dataset),ncol=1)
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  test_indices <- which(folds==i,arr.ind=TRUE)
  testing_dataset <- randomized_data[test_indices, ]
  training_dataset <- randomized_data[-test_indices, ]
  cross_validation_model <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,training_dataset) #Train the Model on 90% of the data
  cv_pred <- predict(cross_validation_model,testing_dataset) #Test on the% of the data
  row_names <- rownames(as.matrix(cv_pred))
  for(j in 1:length(cv_pred))
  {
    fitted_values[as.numeric(row_names[j])] = cv_pred[j] #Store the fitted values in an array
  }
}
actual_values <- network_backup_dataset$Size.of.Backup..GB.
cv_rmse <- sqrt(sum((actual_values-fitted_values)^2)/nrow(network_backup_dataset)) #Calculate the RMSE = 0.7409
################################################################################
#Fitted Values and Actual Values Scatter Plot over Time
library(lattice)
res <- stack(data.frame(Observed = network_backup_dataset$Size.of.Backup..GB., Predicted = fitted_values))
x = data.frame(cbind(network_backup_dataset$Day.of.Week,network_backup_dataset$Backup.Start.Time...Hour.of.Day,network_backup_dataset$Backup.Time..hour.))
res <- cbind(res, x = rep(x$X1, 2))
res <- cbind(res, x = rep(x$X2, 2))
res <- cbind(res, x = rep(x$X3, 2))
#x <- data.frame(network_backup_dataset$Week..)
xyplot(values ~ x, data = res, group = ind, auto.key = TRUE,xlab='Week',ylab='Backup Size in GB')
xyplot(network_backup_dataset$Size.of.Backup..GB. ~ x, data = res, type = c("p","r"), col.line = "red")
################################################################################
#Residuals versus Fitted Values Plot
library('ggplot2')
qplot(y=(actual_values-fitted_values),x=fitted_values,xlab='Fitted Values',ylab='Residuals')
################################################################################
#Cross Validation using in-built function
library('DAAG')
cv_model<-cv.lm(data = network_backup_dataset, form.lm = formula
                (Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.),
                m = 10)
predicted_values <- cv_model$cvpred
actual_values <- network_backup_dataset$Size.of.Backup..GB.
cv_model_inbuilt_rmse <- sqrt(sum((actual_values-predicted_values)^2)/nrow(network_backup_dataset)) #RMSE = 07407
################################################################################