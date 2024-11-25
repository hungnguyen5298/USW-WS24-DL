import praw
import pandas as pd
from datetime import datetime, timezone

reddit = praw.Reddit(
    client_id="y4CG5XvQvivbBO-qFOz4ew",  
    client_secret="eeq3JdB0XKVU1AOb4jj_FF5sJztBCQ",
    user_agent="mein_reddit_scraper",
    user_name="usw_ws24",
    password="ZGC6d37yhQAt_k!")

data = []

for submission in reddit.subreddit('Stocks+investing+finance').new(limit=100):
    submission.comments.replace_more(limit=None)

    comments = [comment.body for comment in submission.comments.list()]

    post_data = {
        'submission_title': submission.title,
        'submission_text': submission.selftext,
        'submission_url': submission.url,
        'upvoted': submission.score,
        'published_at': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'comments': comments
    }
    data.append(post_data)

df = pd.DataFrame(data)

print(df)
df.to_csv("unfiltered_reddit_praw.csv", index=True)
