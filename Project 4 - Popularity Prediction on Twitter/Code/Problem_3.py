import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import linear_model
'''
This function takes the entire hash tag file as the input and calculates all the features
'''
def retrieve_hourly_features(file_id):
    st = datetime.datetime(2016,03,15) #Arbitraray start time
    et = datetime.datetime(2001,03,15) #Arbitrary end time
    hourwise_features = {}
    users_per_hour = {} #Set to store the list of unique users
    file_name = os.path.join(folder_name, train_files[file_id])
    with open(file_name) as tweet_data: #Opening the respective file
        for individual_tweet in tweet_data: #For everyy line in the file
            individual_tweet_data = json.loads(individual_tweet) #Store an individual tweet as JSON object
            individual_time = individual_tweet_data["firstpost_date"] #The time when the tweet was posted
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            #Retrieving the user_id of the user who posted the tweet           
            individual_user_id = individual_tweet_data["tweet"]["user"]["id"] 
            #Retrieving the number of retweets          
            retweet_count = individual_tweet_data["metrics"]["citations"]["total"]
            #Retrieving the followers of the user           
            followers_count = individual_tweet_data["author"]["followers"]
            #Storing the total user user mentions in one tweet            
            user_mention_count = len(individual_tweet_data["tweet"]["entities"]["user_mentions"])
            #Storing the total urls in one tweet
            url_count = len(individual_tweet_data["tweet"]["entities"]["urls"])
            if url_count>0:
                url_count = 1 #If atleast 1 URL, then make it true, else false
            else:
                url_count = 0
            #Retrieving the list count of the user 
            listed_count = individual_tweet_data["tweet"]["user"]["listed_count"]
            if(listed_count == None):
                listed_count = 0
            #Obtaining the total number of 'likes' that a tweet has received            
            favorite_count = individual_tweet_data["tweet"]["favorite_count"]
            #Getting the rank of tweet            
            ranking_score = individual_tweet_data["metrics"]["ranking_score"]
            #Checking if the user is a verified user or not            
            user_verified = individual_tweet_data["tweet"]["user"]["verified"]
            #Inserting a new hour, initilizing features with zeros            
            if modified_time not in hourwise_features:
                hourwise_features[modified_time] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1,'avg_user_mention_count':0,
                'url_count':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}
                users_per_hour[modified_time] = Set([])
            hourwise_features[modified_time]['tweets_count'] += 1
            hourwise_features[modified_time]['retweets_count'] += retweet_count
            hourwise_features[modified_time]['time'] = individual_time.hour
            hourwise_features[modified_time]['avg_user_mention_count'] += user_mention_count
            hourwise_features[modified_time]['url_count'] += url_count
            hourwise_features[modified_time]['avg_favorite_count'] += favorite_count
            if favorite_count > hourwise_features[modified_time]['max_favorite_count']:
                hourwise_features[modified_time]['max_favorite_count'] = favorite_count
            hourwise_features[modified_time]['sum_ranking_score'] += ranking_score
            if individual_user_id not in users_per_hour[modified_time]:
                users_per_hour[modified_time].add(individual_user_id)
                hourwise_features[modified_time]['followers_count'] += followers_count
                hourwise_features[modified_time]['avg_listed_count'] += listed_count
                hourwise_features[modified_time]['user_count'] += 1
                if followers_count > hourwise_features[modified_time]['max_followers']:
                    hourwise_features[modified_time]['max_followers'] = followers_count
                if listed_count > hourwise_features[modified_time]['max_listed_count']:
                    hourwise_features[modified_time]['max_listed_count'] = listed_count
                if  (user_verified):
                    hourwise_features[modified_time]['total_verified_users'] += 1
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
    #with open(train_files[file_id]+'.json', 'wb') as fp:
    #    json.dump(hourwise_features, fp)
    return hourwise_features

'''
This function takes in the hourwise features for each hashtag as the input and computes the Predictor and Label Matrix
The labels are the tweet counts for the next hour
'''
def variables_lables_matrix(hourwise_features):
    start_time = min(hourwise_features.keys()) #Find the start and end time from the data
    end_time = max(hourwise_features.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: #Keep looping from the start time till the end time
        next_hour_tweet_count = 0 #Initialize the label to be zero
        next_hour = cur_hour+timedelta(hours=1) #Go to the next hour
        if next_hour in hourwise_features:
            next_hour_tweet_count = hourwise_features[next_hour]['tweets_count'] #Update the label
        if cur_hour in hourwise_features:
            predictors.append(hourwise_features[cur_hour].values()) #Obtain the predictors
            labels.append([next_hour_tweet_count])
        else: #If a particular hour doesn't exist then initialize with zero
            temp = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':cur_hour.hour,'avg_user_mention_count':0,
                'url_count':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}    
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels
    
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(train_files)):
    hourwise_features = retrieve_hourly_features(i)
    modified_hourwise_features = {}
    for time_value in hourwise_features:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hourwise_features[time_value]
        modified_hourwise_features[cur_hour] = features
    predictors, labels = variables_lables_matrix(modified_hourwise_features)
    predictors = sm.add_constant(predictors)    
    model = sm.OLS(labels, predictors)
    results = model.fit()
    #print(results.summary())
    #with open("linear_regression_problem_3_result"+hashtag_list[i]+".txt", 'wb') as fp:
    #    print >>fp, results.summary()
    '''
    Now all the hashtags were trained with all the fourteen parameters that we had selected.
    X1: sum_ranking_score
    X2:  retweets_count
    X3: tweets_count
    X4: max_listed_count
    X5: total_verified_users
    X6: user_count
    X7: avg_favorite_count
    X8: url_count
    X9: max_favorite_count
    X10: followers_count
    X11: avg_user_mention_count
    X12: time
    X13: avg_listed_count
    X14: max_followers
    From the previous model training we identified the best features and trained new models.
    Too many parameters can lead to over fitting. So we selected the best features for each.    
    #gohawks : X1, X6, X11, X13, X14
    #gopatriots : X1, X8, X11, X13, X14
    #nfl : X3, X11, X13, X14
    #patriots: X1, X5, X6, X10, X11
    #sb49 : X1, X2, X6, X11, X13
    #superbowl : X2, X6, X9, X13, X14
    '''
    if(i==0):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,1],color='r')
        plt.xlabel('Feature : Ranking Score')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,13],color='r')
        plt.xlabel('Feature : User Listed Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,14],color='r')
        plt.xlabel('Feature : Maximum Followers')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)
        plt.close()
        #Model for #gohawks
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,1],predictors[:,6],predictors[:,11],predictors[:,13],predictors[:,14]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary()
    if(i==1):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,1],color='r')
        plt.xlabel('Feature : Ranking Score')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,8],color='r')
        plt.xlabel('Feature : URL Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,11],color='r')
        plt.xlabel('Feature : User Mention Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)        
        plt.close()
        #Model for #gopatriots
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,1],predictors[:,8],predictors[:,11],predictors[:,13],predictors[:,14]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary()
    if(i==2):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,3],color='r')
        plt.xlabel('Feature : Tweet Count in current Hour')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,13],color='r')
        plt.xlabel('Feature : User Listed Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,14],color='r')
        plt.xlabel('Feature : Maximum Followers')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)        
        plt.close()       
        #Model for #nfl
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,3],predictors[:,11],predictors[:,13],predictors[:,14]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary()
    if(i==3):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,1],color='r')
        plt.xlabel('Feature : Ranking Score')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,5],color='r')
        plt.xlabel('Feature : Verified Users')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,10],color='r')
        plt.xlabel('Feature : Followers Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)        
        plt.close()       
        #Model for #patriots
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,1],predictors[:,5],predictors[:,6],predictors[:,10],predictors[:,11]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary()
    if(i==4):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,2],color='r')
        plt.xlabel('Feature : Retweet Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,11],color='r')
        plt.xlabel('Feature : User Mention Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,13],color='r')
        plt.xlabel('Feature : User Listed Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)        
        plt.close()        
        #Model for #sb49
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,1],predictors[:,2],predictors[:,6],predictors[:,11],predictors[:,13]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary() 
    if(i==5):
        #Plotting and storing scatter plots for best three features        
        plt.gca().scatter(labels,predictors[:,2],color='r')
        plt.xlabel('Feature : Retweet Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_1.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,9],color='r')
        plt.xlabel('Feature : Maximum Favorite Tweet Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_2.png"
        plt.savefig(imageName)
        
        plt.gca().scatter(labels,predictors[:,13],color='r')
        plt.xlabel('Feature : User Listed Count')
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtag_list[i] + "_best_feature_3.png"
        plt.savefig(imageName)        
        plt.close()        
        #Model for #superbowl    
        predictors = np.transpose(np.asarray([predictors[:,0],predictors[:,2],predictors[:,6],predictors[:,9],predictors[:,13],predictors[:,14]]))
        model = sm.OLS(labels, predictors)
        results = model.fit()
        with open("linear_regression_problem_3_best_result"+hashtag_list[i]+".txt", 'wb') as fp:
            print >>fp, results.summary()
    '''    
    labels = [int(j[0]) for j in labels]
    clf_rf = RandomForestClassifier(n_estimators=1,oob_score='true')
    clf_rf = clf_rf.fit(predictors, np.asarray(labels))
    y_pred_rf = clf_rf.score(predictors, np.asarray(labels))
    '''