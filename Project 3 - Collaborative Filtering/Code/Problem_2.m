%Creating indices for 10 Fold Cross Validation
filename = 'u.data';
delimiterIn = ('\t');
u= dlmread(filename, delimiterIn);
Indices = crossvalind('Kfold', 100000, 10);
err = zeros(10,1,3); %This array will store the absolute error for the 10 folds
K = [10, 50, 100];
for kk=1:3
    for i=1:10
        test = zeros(10000,5);
        R_train = NaN(943,1682);
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
        [U,V] = wnmfrule(R_train,K(kk)); %Performing NNMF
        UV = U*V;
        for j=1:10000
            test(j,4) = UV(test(j,1),test(j,2)); %Storing the predicted values for the test data
            test(j,5) = abs(test(j,3) - test(j,4)); %Calculating the absolte difference between predicted and actual values
        end
        error = sum(test(:,5)); %Summing up the absolute errors for each fold
        error = error/10000;
        err(i,1,kk) = error;
    end
end

% err
% 
% err(:,:,1) =
% 
%     0.7902
%     0.8048
%     0.7941
%     0.7916
%     0.7910
%     0.8041
%     0.8021
%     0.7790
%     5.5221
%     4.4543
% 
% 
% err(:,:,2) =
% 
%     0.8927
%     0.9157
%     0.8985
%     0.9007
%     0.8945
%     0.8797
%     0.9079
%     0.9050
%     0.8982
%     0.8789
% 
% 
% err(:,:,3) =
% 
%     0.9094
%     0.9052
%     0.9098
%     0.9041
%     0.8966
%     0.9126
%     0.9315
%     0.8848
%     0.9018
%     0.9054