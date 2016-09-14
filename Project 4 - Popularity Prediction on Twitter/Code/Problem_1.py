import os
import json
import datetime
from sets import Set
from datetime import timedelta
import matplotlib.pyplot as plt

'''
This function calcluates the asked statistics for each of the hashtag.
It returns a series of statistics.
'''
def calculate_statistics(file_id):
    st = datetime.datetime(2016,03,15) #Arbitraray start time
    et = datetime.datetime(2001,03,15) #Arbitrary end time
    unique_user_list = Set([]) #Set to store the list of unique users
    total_followers = 0.0
    total_retweets = 0.0 
    total_tweets = 0.0
    total_hours = 0.0
    hour = {}
    file_name = os.path.join(folder_name, train_files[file_id]) 
    with open(file_name) as tweet_data: #Opening the respective file
        for individual_tweet in tweet_data: #For everyy line in the file
            individual_tweet_data = json.loads(individual_tweet) #Store an individual tweet as JSON object
            individual_user_id = individual_tweet_data["tweet"]["user"]["id"] #Retrieving the user_id of the user who posted the tweet
            if individual_user_id not in unique_user_list: #Add the user_id only if it is not in the list of all users and update the followers count
                total_followers += individual_tweet_data["author"]["followers"]
                unique_user_list.add(individual_user_id)
            total_retweets += individual_tweet_data["metrics"]["citations"]["total"] #Update the retweet count
            individual_time = individual_tweet_data["firstpost_date"] #The time when the tweet was posted
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
            total_tweets = total_tweets + 1 #Update the number of tweets
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            if modified_time not in hour:
                hour[modified_time] = {'hour_tweets_count':0}
            hour[modified_time]['hour_tweets_count'] += 1 
    total_hours = int((et - st).total_seconds()/3600 + 0.5) #Calculate the total hours from the time of first tweet till last tweet   
    return total_followers, total_retweets, total_tweets, total_hours, len(unique_user_list), hour

def  histogram_one_hour_bins(file_id, hour):
    first_tweet_time = min(hour.keys())
    last_tweet_time = max(hour.keys())
    tweets_by_hour = []
    current_time = first_tweet_time
    while current_time <= last_tweet_time:
        if current_time in hour:
            tweets_by_hour.append(hour[current_time]["hour_tweets_count"])
        else:
            tweets_by_hour.append(0)
        current_time += timedelta(hours=1)        
    #Plotting the histogram
    plt.figure(figsize=(20, 8))
    plt.title("Number of Tweets per Hour plot for " + hashtag_list[file_id])
    plt.ylabel("Number of Tweets")
    plt.xlabel("Hours Elapsed")
    plt.bar(range(len(tweets_by_hour)), tweets_by_hour)
    plt.show()
    
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(train_files)):
    total_followers, total_retweets, total_tweets, total_hours, total_users, hour = calculate_statistics(i)
    print "Average number of tweets per hour for",  hashtag_list[i], "are", total_tweets/total_hours
    print "Average number of followers of users posting tweets for",  hashtag_list[i], "are", total_followers/total_users
    print "Average number of retweets for", hashtag_list[i], "are", total_retweets/total_tweets
    modified_hour = {}
    for time_value in hour:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hour[time_value]
        modified_hour[cur_hour] = features    
    if(i == 2 or i == 5):
        histogram_one_hour_bins(i, modified_hour)