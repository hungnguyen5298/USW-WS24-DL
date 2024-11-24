import praw
import pandas as pd

# Reddit API-Konfiguration
reddit = praw.Reddit(
    client_id="usw_ws24",  # Ersetze mit deiner client_id
    client_secret="eeq3JdB0XKVU1AOb4jj_FF5sJztBCQ",  # Ersetze mit deinem client_secret
    user_agent="mein_reddit_scraper",
)

print(reddit.user.me())
