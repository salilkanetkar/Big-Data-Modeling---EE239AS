#Problem2B
#Using the random forest function for cross validation

################################################################################
library(randomForest)
rf_rmse_all = NULL
rf_ntree = NULL
rf_nodesize = NULL
ns=4647
actual_values <- network_backup_dataset$Size.of.Backup..GB.
for(i in 1:300)
{
  nt = 20+i-1
  rf_model <- randomForest(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
                           data = network_backup_dataset,ntree=nt,nodesize=ns)#Training the Model
  rf_ntree = append(rf_ntree,nt)
  rf_nodesize = append(rf_nodesize,ns)
  ns = ns - 20
  rf_predicted_values <- rf_model$predicted
  rf_rmse <- sqrt(sum((actual_values-rf_predicted_values)^2)/nrow(network_backup_dataset))
  rf_rmse_all = append(rf_rmse_all,rf_rmse)
}
################################################################################
plot(rf_rmse_all, ylab="RMSE")
plot(y=rf_rmse_all,x=rf_ntree, ylab="RMSE",xlab ="No of Trees")
plot(y=rf_rmse_all,x=rf_nodesize, ylab="RMSE",xlab ="Node Size")
################################################################################