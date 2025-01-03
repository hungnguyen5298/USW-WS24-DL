from eventregistry import EventRegistry, QueryArticlesIter
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Initialize EventRegistry with API key
er = EventRegistry(apiKey='008ffa8f-3bc2-4fbf-8aad-1d98dcc66279')  # Thay YOUR_API_KEY bằng API key của bạn

# Tính toán thời gian 1 giờ trước
one_hour_ago = datetime.now() - timedelta(hours=1)
one_hour_ago = one_hour_ago.strftime('%Y-%m-%dT%H:%M:%SZ')  # Định dạng thời gian theo UTC

# Đặt tham số tìm kiếm
query = {
    "$query": {
        "$and": [
            {
                "$or": [
                    {"keyword": "Apple", "keywordLoc": "body"},
                    {"keyword": "AAPL", "keywordLoc": "body"},
                    {"keyword": "iPhone", "keywordLoc": "body"},
                    {"keyword": "MacBook", "keywordLoc": "body"},
                    {"keyword": "iPad", "keywordLoc": "body"}
                ]
            },
            {
                "date": {
                    "$gte": one_hour_ago  # Lấy bài báo trong vòng 1 giờ qua
                }
            },
            {"lang": "eng"}  # Ngôn ngữ tiếng Anh
        ]
    }
}

# In ra truy vấn để kiểm tra xem có gì sai không
print("Query being used:", query)

# Thực hiện truy vấn
q = QueryArticlesIter.initWithComplexQuery(query)

# Lưu danh sách bài báo
articles = []
berlin_tz = pytz.timezone('Europe/Berlin')  # Múi giờ Berlin

# Kiểm tra nếu có bài viết trả về
articleList = q.execQuery(er, maxItems=100)

if not articleList:
    print("No articles found.")
else:
    for article in articleList:
        # Kiểm tra nếu bài báo có trường 'date' (thời gian đăng)
        if 'date' in article:
            # Lấy thời gian đăng bài và chuyển đổi sang múi giờ Berlin
            published_utc = datetime.strptime(article.get("date"), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.utc)
            published_berlin = published_utc.astimezone(berlin_tz).strftime('%H:%M:%S')  # Lấy giờ

            # Thêm bài báo vào danh sách
            articles.append({
                "Title": article.get("title"),
                "Description": article.get("body"),
                "Publisher": article.get("source", {}).get("title"),
                "URL": article.get("url"),
                "Published on": article.get("date"),
                "Timestamp (Berlin)": published_berlin
            })

    # Chuyển danh sách bài báo thành DataFrame
    df = pd.DataFrame(articles)

    # Kiểm tra nếu DataFrame có dữ liệu
    if df.empty:
        print("No articles returned.")
    else:
        # Loại bỏ các bài báo trùng tiêu đề
        df = df.drop_duplicates(subset=["Title"])

        # Chuyển đổi 'Published on' sang datetime để sắp xếp
        df['Published on'] = pd.to_datetime(df['Published on'], errors='coerce')

        # Loại bỏ các bài báo có lỗi định dạng thời gian
        df = df.dropna(subset=['Published on'])

        # Sắp xếp bài báo theo thời gian từ mới nhất đến cũ nhất
        df = df.sort_values(by='Published on', ascending=False)

        # Hiển thị DataFrame
        print(df)

        # Lưu DataFrame vào file CSV
        df.to_csv("filtered_recent_apple_articles.csv", index=False)
