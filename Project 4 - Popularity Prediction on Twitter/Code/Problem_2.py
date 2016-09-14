import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm

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
            #Inserting a new hour, initilizing features with zeros            
            if modified_time not in hourwise_features:
                hourwise_features[modified_time] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1}
                users_per_hour[modified_time] = Set([])
            hourwise_features[modified_time]['tweets_count'] += 1
            hourwise_features[modified_time]['retweets_count'] += retweet_count
            hourwise_features[modified_time]['time'] = individual_time.hour
            if individual_user_id not in users_per_hour[modified_time]: #If a user is not added, then add it
                users_per_hour[modified_time].add(individual_user_id)
                hourwise_features[modified_time]['followers_count'] += followers_count
                if followers_count > hourwise_features[modified_time]['max_followers']:
                    hourwise_features[modified_time]['max_followers'] = followers_count
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
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
            temp = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':cur_hour.hour}    
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels
    
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(train_files)):
    i=0    
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
    with open("linear_regression_result_problem_2_"+hashtag_list[i]+".txt", 'wb') as fp:
        print >>fp, results.summary()