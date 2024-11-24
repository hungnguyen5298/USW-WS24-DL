import praw
import pandas as pd

# Reddit API-Konfiguration
reddit = praw.Reddit(
    client_id="y4CG5XvQvivbBO-qFOz4ew",
    client_secret="eeq3JdB0XKVU1AOb4jj_FF5sJztBCQ",
    user_agent="mein_reddit_scraper",
    user_name="usw_ws24",
    password="ZGC6d37yhQAt_k!")

# Subreddit auswählen und nach Beiträgen suchen
subreddit = reddit.subreddit('Stocks+investing+finance')  # Subreddit: 'Stocks'
search_query = "Apple OR AAPL OR iPhone"  # Suchbegriffe
posts = subreddit.search(search_query, sort="new", limit=100)  # Neueste Beiträge zu Apple

# Liste zum Speichern der Daten
data = []

# Posts durchgehen und Daten extrahieren
for submission in posts:
    title = submission.title
    content = submission.selftext if submission.selftext else "No content"
    url = submission.url
    upvotes = submission.score

    # Kommentare extrahieren
    submission.comments.replace_more(limit=0)
    comments = [comment.body for comment in submission.comments.list()]

    data.append({
        "Title": title,
        "Content": content,
        "URL": url,
        "Upvotes": upvotes,
        "Comments": "; ".join(comments)
    })

# DataFrame erstellen
df = pd.DataFrame(data)

# DataFrame in CSV speichern
df.to_csv("apple_discussions_in_stocks.csv", index=False)

print("Daten wurden erfolgreich in 'apple_discussions_in_stocks.csv' gespeichert.")
