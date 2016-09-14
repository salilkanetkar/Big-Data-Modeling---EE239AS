
#EE239 - Problem 1 

#converting days of the week into integer values
list_of_days <- levels(network_backup_dataset$Day.of.Week)[c(2,6,7,5,1,3,4)]
factoring_days <- factor(network_backup_dataset$Day.of.Week,list_of_days)
numeric_day <- as.numeric(factoring_days)

#appending the numberic day column to original dataset
network_backup_dataset$numeric.day <- numeric_day

#calculating the day number depending on the week number
day_number = ((network_backup_dataset$Week.. -1)*7)+network_backup_dataset$numeric.day

#appending the day number to original dataset
network_backup_dataset$day.number <- day_number

#create a new dataset only for 20 days

days <- c(21:40)

new_numeric_day = NULL
new_workflow_id = NULL
new_file_name = NULL
new_size_of_backup = NULL
new_day_number = NULL

for(j in 1:18588) #select values that fall within 20 days
{
  if(network_backup_dataset$day.number[j] %in% days)
  {
  new_numeric_day = c(new_numeric_day, network_backup_dataset$numeric.day[j])
  new_workflow_id = c(new_workflow_id, network_backup_dataset$Work.Flow.ID[j])
  new_file_name = c(new_file_name, network_backup_dataset$File.Name[j])
  new_size_of_backup = c(new_size_of_backup, network_backup_dataset$Size.of.Backup..GB.[j])
  new_day_number = c(new_day_number, network_backup_dataset$day.number[j])
  }
}

#newly created dataset that is to be used for further processing
new_dataset = data.frame(new_numeric_day, new_day_number, new_workflow_id,
                         new_file_name, new_size_of_backup)

#sort the new dataset with respect to workflow id
new_dataset = new_dataset[order(new_dataset$new_workflow_id),]

#sum up the backup sizes for each day
i = 1;
backup_size = matrix(0,100)
for( j in 1:3537)
{
  backup_size[i] = backup_size[i]+new_dataset$new_size_of_backup[j];
  
  if(new_dataset$new_day_number[j] != new_dataset$new_day_number[j+1])
  {
    i = i+1
  }
}
backup_size[i] = backup_size[i]+new_dataset$new_size_of_backup[3538]


#arrange the data according to workflow and day of week
workflow = matrix(0,100);
day_of_week = matrix(0,100);
i = 1;
for( j in 1:5)
{
  x=1;
  for( k in 1:20)
  {
    workflow[i] = j-1;
    day_of_week[i]  = x;
    x = x+1;
    i = i+1;
  }
}


#now to be plotted
dataset_final <- data.frame(day_of_week , workflow , backup_size)

install.packages("ggplot2")
library(ggplot2)

#plot the graph
ggplot(dataset_final, aes(x=dataset_final$day_of_week, y=dataset_final$backup_size, colour=workflow))+ 
  geom_line(aes(group=dataset_final$workflow)) + 
  labs(x="Number of Days", y="Size of Backup (GB)")


