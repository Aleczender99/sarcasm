import tweepy
import json

import TwitterCredentials

# Authenticate with the Twitter API using Tweepy
auth = tweepy.Client(TwitterCredentials.Bearer)

# Define a function to fetch and save tweets
def fetch_tweets(query, count):
    """
    This function takes a search query and the number of tweets to fetch as input, and saves the
    fetched tweets to a SQLite database after cleaning the text using the clean_text function.
    """
    tweets = auth.search_recent_tweets(query=query, max_results=count,
                                       tweet_fields=["created_at", "public_metrics", "text"],
                                       user_fields=["name", "username"])
    with open("tweets.json", "w") as f:
        for tweet in tweets.data:
            tweet_dict = {"tweet": tweet.text}
            tweet_json = json.dumps(tweet_dict)
            f.write(tweet_json + "\n")


# Fetch and save tweets for the query "Python"
fetch_tweets("#sarcasm", 100)
