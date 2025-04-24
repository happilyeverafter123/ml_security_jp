from dotenv import load_dotenv
import os
import tweepy
import json
import numpy as np
import pandas as pd

load_dotenv()

BEARER_TOKEN = os.environ.get('BEARER_TOKEN')
CONSUMER_KEY = os.environ.get('CONSUMER_KEY')
CONSUMER_SECRET = os.environ.get('CONSUMER_SECRET')
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.environ.get('ACCESS_TOKEN_SECRET')

client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

search_term = 'exploit OR vulnerability OR CVE OR "zero day" OR "0day" OR "zero-day" OR "0 day" OR "zero day" OR "0 day" OR "exploit kit" OR "exploit code" OR "exploit database" OR "exploit pack" OR "exploit server" OR "exploit tool" OR "exploit tools" OR "exploit vulnerability" OR "exploiting vulnerability" OR "exploiting zero day" OR "exploiting 0day"'
                
oldest_tweet = None

TempDict = []

counter = 0

response = client.search_recent_tweets(
    query=search_term,
    max_results=10,
    tweet_fields=["id", "text", "created_at"]
)

for tweet in response.data:
    print(tweet.text)

# for x in range(10):
#     public_tweets = client.search_recent_tweets(query=search_term,
#                                                 max_results=100,
#                                                 pagination_token=oldest_tweet
#                                                 tweet_fields=["id", "text", "created_at"]
#                                                 until_id=oldest_tweet
# )
#     
#     for tweet in public_tweets['data']:
#         TempDict.append(tweet)
#         counter += 1
#         oldest_tweet = tweet['id']
#         print("Quoted tweet, skipping...")
#         
#     print(f"Fetched {counter} tweets")
