#Problem3
################################################################################
#Piece Wise Linear Regression
#Subsetting the dataset on the basis of work flows
work_flow0_subset = subset(network_backup_dataset, Work.Flow.ID == "work_flow_0")
work_flow1_subset = subset(network_backup_dataset, Work.Flow.ID == "work_flow_1")
work_flow2_subset = subset(network_backup_dataset, Work.Flow.ID == "work_flow_2")
work_flow3_subset = subset(network_backup_dataset, Work.Flow.ID == "work_flow_3")
work_flow4_subset = subset(network_backup_dataset, Work.Flow.ID == "work_flow_4")
#Training the model and calculating the RMSE for each work flow
piece_0 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=subset(network_backup_dataset,Work.Flow.ID == "work_flow_0"))
piece_0_rmse <- sqrt(sum((work_flow0_subset$Size.of.Backup..GB.-piece_0$fitted.values)^2)/nrow(network_backup_dataset[piece_workflow4,]))
piece_1 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=subset(network_backup_dataset,Work.Flow.ID == "work_flow_1"))
piece_1_rmse <- sqrt(sum((work_flow1_subset$Size.of.Backup..GB.-piece_1$fitted.values)^2)/nrow(network_backup_dataset[piece_workflow4,]))
piece_2 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=subset(network_backup_dataset,Work.Flow.ID == "work_flow_2"))
piece_2_rmse <- sqrt(sum((work_flow2_subset$Size.of.Backup..GB.-piece_2$fitted.values)^2)/nrow(network_backup_dataset[piece_workflow4,]))
piece_3 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=subset(network_backup_dataset,Work.Flow.ID == "work_flow_3"))
piece_3_rmse <- sqrt(sum((work_flow3_subset$Size.of.Backup..GB.-piece_3$fitted.values)^2)/nrow(network_backup_dataset[piece_workflow4,]))
piece_4 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=subset(network_backup_dataset,Work.Flow.ID == "work_flow_4"))
piece_4_rmse <- sqrt(sum((work_flow4_subset$Size.of.Backup..GB.-piece_4$fitted.values)^2)/nrow(network_backup_dataset[piece_workflow4,]))
piece_total_rmse <- c(piece_0_rmse,piece_1_rmse,piece_2_rmse,piece_3_rmse,piece_4_rmse)
qplot(x=unique(network_backup_dataset$Work.Flow.ID),y=piece_total_rmse)
mean(piece_total_rmse)
plot(piece_total_rmse,xlab="Work Flows",ylab="RMSE Values",type='h')

################################################################################

#PolynomialRegression
ds <- network_backup_dataset_modified
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
    cv_poly_model <- lm(Size.of.Backup..GB. ~ polym((Day.of.Week), Backup.Start.Time...Hour.of.Day, Backup.Time..hour.,degree = a,raw = TRUE),data=training_dataset)
    training_dataset <- randomized_data[test_indices, ]
    cv_pred <- predict(cv_poly_model,training_dataset)
    name <- rownames(as.matrix(cv_pred))
    for(j in 1:length(cv_pred))
    {
      fitted_values[as.numeric(name[j])] = cv_pred[j] #String the fitted values
    }
  }
  actual_values <- ds$Size.of.Backup..GB.
  rmse <- sqrt(mean((actual_values-fitted_values)^2))
  poly_rmse_all <- append(poly_rmse_all,rmse)
}
plot(poly_rmse_all[1:8],xlab="Degree of Polynomial",ylab="RMSE")
plot(poly_rmse_all,xlab="Degree of Polynomial",ylab="RMSE")
################################################################################