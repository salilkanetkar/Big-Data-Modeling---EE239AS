%Creating an empty matrix R initialized with zeros
filename = 'u.data';
delimiterIn = ('\t');
u= dlmread(filename, delimiterIn);
R = NaN(943,1682);
num = size(u,1);
K = [10, 50, 100];
%Storing the values in R from the Dataset
for i=1:num
    R(u(i,1),u(i,2)) = u(i,3);
end
%Creating indices for 10 Fold Cross Validation
Indices = crossvalind('Kfold', 100000, 10);
%Creating an empt matrix to store the predicted values
R_Predicted = NaN(943,1682,3); 
% threshold vector would hold different 50  equidistant values of threshold from 0.1 to 5.0
threshold = linspace(0.1, 5.0, 50); 
% Create precision and recall vectors for each value of threshold
precision_vector = zeros(50,1,3);
recall_vector = zeros(50,1,3);
%Calculating predicted values using cross-validation
for kk=1:3
    for i=1:10
        test = zeros(10000,3); %Will store 10% of the data
        R_train = NaN(943,1682); %Will store 90% of the data
        W_train = NaN(943,1682);
        k = 1;
        for j=1:100000
            if(Indices(j) ~= i)
                R_train(u(j,1),u(j,2)) = u(j,3); %Creating the R Matrix from the Training Data
                W_train(u(j,1),u(j,2)) = 1;
            else
                test(k,1) = u(j,1);	%Stroring the Test Data
                test(k,2) = u(j,2); %in a separate 2D Matrix
                test(k,3) = u(j,3); %called the test
                k=k+1;
            end
        end
        [U,V] = wnmfrule(R_train,K(kk)); %Performing NNMF with K=100
        UV = U*V;
        for j=1:10000
            R_Predicted(test(j,1),test(j,2),kk) = UV(test(j,1),test(j,2)); %The predicted value is now being stored in the R_Predicted
        end
    end
end
% Loop to calculate value of precision and recall for each threshold
for kk=1:3
    for i=1:50
        precision_vector(i,1,kk)=length(find((R_Predicted(:, :,kk)>threshold(i)) & (R>3)))/length(find(R_Predicted(:, :,kk)>threshold(i)));
        recall_vector(i,1,kk)=length(find((R_Predicted(:, :,kk)>threshold(i)) & (R>3)))/length(find(R>3));
    end
end


%Plot precision over recall values for different values of k
figure;
plot(recall_vector(:,1,1), precision_vector(:,1,1),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Recall for k=10')
xlabel('Recall')
ylabel('Precision')

figure;
plot(recall_vector(:,1,2), precision_vector(:,1,2),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Recall for k=50')
xlabel('Recall')
ylabel('Precision')

figure;
plot(recall_vector(:,1,3), precision_vector(:,1,3),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Recall for k=100')
xlabel('Recall')
ylabel('Precision')

% Plot precision over threshold values for different values of k
figure;
plot(threshold(:), precision_vector(:,1,1),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Threshold for k=10')
xlabel('Threshold')
ylabel('Precision')

figure;
plot(threshold(:), precision_vector(:,1,2),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Threshold for k=50')
xlabel('Threshold')
ylabel('Precision')

figure;
plot(threshold(:), precision_vector(:,1,3),'Marker','o','MarkerFaceColor','black')
title('Precision v/s Threshold for k=100')
xlabel('Threshold')
ylabel('Precision')


% Plot recall over threshold values for different values of k
figure;
plot(threshold(:), recall_vector(:,1,1),'Marker','o','MarkerFaceColor','black')
title('Recall v/s Threshold for k=10')
xlabel('Threshold')
ylabel('Recall')

figure;
plot(threshold(:), recall_vector(:,1,2),'Marker','o','MarkerFaceColor','black')
title('Recall v/s Threshold for k=50')
xlabel('Threshold')
ylabel('Recall')

figure;
plot(threshold(:), recall_vector(:,1,3),'Marker','o','MarkerFaceColor','black')
title('Recall v/s Threshold for k=100')
xlabel('Threshold')
ylabel('Recall')
