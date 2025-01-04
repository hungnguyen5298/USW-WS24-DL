import os
import praw
import pandas as pd
from datetime import datetime, timezone
from pytz import timezone, utc

# Reddit API-Konfiguration
reddit = praw.Reddit(
    client_id="y4CG5XvQvivbBO-qFOz4ew",
    client_secret="eeq3JdB0XKVU1AOb4jj_FF5sJztBCQ",
    user_agent="mein_reddit_scraper",
    user_name="usw_ws24",
    password="ZGC6d37yhQAt_k!")

# Subreddit auswählen und nach Beiträgen suchen
subreddit = reddit.subreddit('stocks+investing+finance+stockmarket+wallstreetbets+trading')  # Subreddits kombinieren
search_query = "Apple OR AAPL OR iPhone OR iPad OR Macbook OR AppleInc"  # Suchbegriffe
posts = subreddit.search(search_query, sort="new", limit=200)  # Neueste Beiträge abrufen

# Liste zum Speichern der Daten
data = []

# Beiträge durchgehen und Daten extrahieren
for submission in posts:
    try:
        title = submission.title
        content = submission.selftext if submission.selftext else "No content"
        url = submission.url
        upvotes = submission.score

        # Kommentare extrahieren
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list()]

        # Veröffentlichungsdatum konvertieren
        utc_time = datetime.fromtimestamp(submission.created_utc, tz=utc)
        germany_tz = timezone('Europe/Berlin')
        germany_time = utc_time.astimezone(germany_tz)
        published_at = germany_time.strftime('%Y-%m-%d %H:%M:%S')

        # Beitrag hinzufügen
        data.append({
            "Title": title,
            "Content": content,
            "URL": url,
            "Upvotes": upvotes,
            "Published_at": published_at,
            "Comments": "; ".join(comments)
        })

    except Exception as e:
        print(f"Fehler bei Beitrag {submission.id}: {e}")

# DataFrame erstellen
if data:
    df = pd.DataFrame(data)

    # Ordner erstellen, falls nicht vorhanden
    os.makedirs("../project_raw_data", exist_ok=True)

    # Daten in CSV speichern
    file_path = os.path.join("../project_raw_data", "filtered_reddit_praw.csv")
    df.to_csv(file_path, index=False, encoding="utf-8")

    print(f"Die Daten wurden erfolgreich in {file_path} gespeichert.")
else:
    print("Keine Daten gefunden.")