import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="y4CG5XvQvivbBO-qFOz4ew",  
    client_secret="eeq3JdB0XKVU1AOb4jj_FF5sJztBCQ",
    user_agent="mein_reddit_scraper",
    user_name="usw_ws24",
    password="ZGC6d37yhQAt_k!")

data = []

for submission in reddit.subreddit("stocks").new(limit=100):
    submission.comments.replace_more(limit=None)

    comments = [comment.body for comment in submission.comments.list()]

    post_data = {
        'submission_title': submission.title,
        'submission_text': submission.selftext,
        'submission_url': submission.url,
        'comments': comments
    }
    data.append(post_data)

df = pd.DataFrame(data)

print(df)
df.to_csv("reddit_posts_comments.csv", index=True)

