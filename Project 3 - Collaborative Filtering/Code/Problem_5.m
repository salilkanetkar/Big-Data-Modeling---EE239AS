%Creating indices for 10 Fold Cross Validation
Indices = crossvalind('Kfold', 100000, 10);

predicted = NaN(943,1682,3);
i = 1;

K = [10, 50, 100];

for kk=1:3

    for i=1:10
    test = zeros(10000,5);
    R_train = NaN(943,1682);
    W_train = NaN(943,1682);
    k = 1;
    for j=1:100000
        if(Indices(j) ~= i)
            W_train(u(j,1),u(j,2)) = u(j,3); %Creating the W Matrix from the Training Data
            R_train(u(j,1),u(j,2)) = 1;
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
        predicted(test(j,1),test(j,2),kk) = UV(test(j,1),test(j,2)); %storing predicted values in predicted matrix
    end
    
    end
    
end

%For each user in the predicted matrix, 
%select all the 1682 movies and sort them in descending order
%then run a for loop of L:1 to 20 and select that many movies for that user
%and store into recommended movies. so that means recommended movies will
%be formed 20 times for each user.
%according to that recommended movies list, calculate all the tp,tn,fp,fn
%and precision and everything
%so finally u have mean precision values for each L.
%plot that

threshold = 0.4; %found from previous question
precision = zeros(943,3); %943 users
hit_rate = zeros(943,20,3); %943 users and 20 L and 3 values of k
fa_rate = zeros(943,20,3); %943 users and 20 L and 3 values of k


for m=1:3 %3 values of k
    for i=1:943 %no. of users
    
    %sort the movies for ith user in descending order based on the ratings
    [~, indices] = sort(predicted(i,:,m),'descend');
    
    %now we have to pick top L movies from this set
    for L=1:20
        top_movies = zeros(943,L,3); %No. of users x L
        count = 1; %to keep track of whether L movies have been selected or not
       
        for t=1:size(indices,2)
            %if ith user has given rating to t movie then only consider
            %the movie
            if R_train(i,indices(t))== 1
                %then add it to list of top movies
                top_movies(i,count,m) = indices(t);
                count = count+1;
            end
            
            %if we have found top L movies,
            if count == L+1
                
                %now that we have top L movies, calculate the tp, tn, fp,
                %fn
                tp = length(find((predicted(i,top_movies(i,:,m),m)> threshold) & (W_train(i,top_movies(i,:,m))>3)));
                tn = length(find((predicted(i,top_movies(i,:,m),m)<= threshold) & (W_train(i,top_movies(i,:,m))<=3)));
                fp = length(find((predicted(i,top_movies(i,:,m),m)> threshold) & (W_train(i,top_movies(i,:,m))<=3)));
                fn = length(find((predicted(i,top_movies(i,:,m),m)<= threshold) & (W_train(i,top_movies(i,:,m))>3)));
               
                %if L is 5, then calculate the precision for the ith user
                if L==5
                    precision(i,m) = tp/length(find(predicted(i, top_movies(i,:,m),m)> threshold));
                end
                
                %Find the hit rate for ith user and L top movies
                if tp==0 && fn==0
                   hit_rate(i,L,m) = 0; %convert NaN to 0
                else
                    hit_rate(i,L,m) = tp/(tp+fn);
                end
                
                %Find the false-alarm rate
                if fp==0 && tn==0
                    fa_rate(i,L,m) = 0; %convert NaN to 0
                else
                    fa_rate(i,L,m) = fp/(fp+tn);
                end
                
                break;    
            
            end
        end
    end
    end 
end

%Now calculate the mean hit rate and false alarm rate for all users
mean_hr = zeros(20,3); %25 values of L
mean_fa = zeros(20,3);

average_precision = zeros(3);
%average precision for when L=5
for m=1:3
    average_precision(m) =  mean(precision(:,m));
end

%mean hit rate and false alarm rate for each value of L
for m=1:3
    for L=1:20
        mean_hr(L,m) = mean(hit_rate(:,L,m));
        mean_fa(L,m) = mean(fa_rate(:,L,m));
    end
end


%Graph of L vs Average Hit Rate
 plot((1:20),mean_hr(:,1),'r',(1:20),mean_hr(:,2),'b',(1:20),mean_hr(:,3),'g')
 xlabel('L')
 ylabel('Average Hit Rate')
 legend('k = 10', 'k = 50', 'k = 100')

%Graph of L vs Average False Alarm Rate 
 plot((1:20),mean_fa(:,1),'r',(1:20),mean_fa(:,2),'b',(1:20),mean_fa(:,3),'g')
 xlabel('L')
 ylabel('Average False Alarm Rate')
 legend('k = 10', 'k = 50', 'k = 100')

%Graph of Average False Alarm Rate vs Average Hit Rate
plot(mean_fa(:,1),mean_hr(:,1),'r',mean_fa(:,2),mean_hr(:,2),'b',mean_fa(:,3),mean_hr(:,3),'g')
 xlabel('Average False Alarm Rate')
 ylabel('Average Hit Rate')
 legend('k = 10', 'k = 50', 'k = 100')



