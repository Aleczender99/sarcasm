import tweepy
import time
import TwitterCredentials
from langdetect import detect
import json

# Authenticate with the Twitter API using Tweepy
auth = tweepy.Client(TwitterCredentials.Bearer)


def fetch_local(query):
    tweets = []

    with open(r"C:\Users\ciaus\PycharmProjects\licenta\tweets.json", 'r', encoding='utf-8') as file:
        # Load JSON data from file
        tweet_list = json.load(file)

    for tweet in tweet_list['data']:
        tweets.append(tweet)

    return tweets


# Fetch and save tweets
def fetch_tweets(query):
    tweets = []

    # Calculate the timestamp for an hour ago
    one_hour_ago = time.time() - 3600  # Subtract 3600 seconds (1 hour)
    one_hour_ago_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(one_hour_ago))

    tweet_list = auth.search_recent_tweets(query=query, max_results=100, tweet_fields='referenced_tweets',
                                           start_time=one_hour_ago_iso)

    for tweet in tweet_list.data:
        if tweet.referenced_tweets is None:
            if detect(tweet.text) == 'en':
                tweets.append(tweet.data)

    return tweets


# Stream Tweets
class TweetStream(tweepy.StreamingClient):
    def on_connect(self):
        print('Connected')

    def on_tweet(self, tweet):
        if tweet.referenced_tweets is None:
            print(tweet.text)


def stream_tweets(query):
    stream = TweetStream(bearer_token=TwitterCredentials.Bearer)
    stream.add_rules(tweepy.StreamRule(query))
    stream.filter(tweet_fields='referenced_tweets')


# Respond to the tweet
def respond_to_tweet(response_text, tweet_id):
    api = tweepy.API(auth)
    api.update_status(status=response_text, in_reply_to_status_id=tweet_id, auto_populate_reply_metadata=True)
    print(f"Successfully replied to tweet ID: {tweet_id}")
