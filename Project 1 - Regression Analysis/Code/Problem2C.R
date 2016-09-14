################################################################################
#EE239-Problem2c
library("neuralnet")

nn_rmse_all = NULL

#actual values of backup size as given in the dataset
actual_values <- network_backup_dataset$Size.of.Backup..GB.

#'Day of week' is a categorical parameter, hence to be converted into quantitative form
numeric_value <- model.matrix(~Size.of.Backup..GB.+ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour., data=network_backup_dataset)

#Fit the neural network model
nn_model <- neuralnet(Size.of.Backup..GB. ~ Day.of.WeekMonday+
                        Day.of.WeekTuesday +
                         Day.of.WeekWednesday +
                         Day.of.WeekThursday +
                         Day.of.WeekSaturday +
                          Day.of.WeekSunday
                         +Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
                         data = numeric_value,hidden=6, threshold=0.1)                        
#results yielded by neural network
nn_predicted_values <- cbind(nn_model$net.result[[1]])
#calculating the root mean square based on actual values and values obtained from neural network  
nn_rmse <- sqrt(sum((actual_values-nn_predicted_values)^2)/nrow(network_backup_dataset))
plot(nn_model)
################################################################################