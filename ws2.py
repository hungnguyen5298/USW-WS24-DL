import time
import httpx
import pandas as pd

base_url = 'https://reddit.com'
endpoint = '/r/stocks'
category = '/hot'

url = base_url + endpoint + category + ".json"
after_post_id = None

dataset = []

for _ in range(5):
    params = {
        'limit': 100,
        't': 'year',
        'after': after_post_id,
    }
    response = httpx.get(url, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception('Failed to fetch data')

    json_data = response.json()

    dataset.extend(json_data['data'] for sec in json_data['data']['children'])

    after_post_id = json_data['data']['after']
    time.sleep(0.5)

df = pd.DataFrame(dataset)
df.to_csv('reddit_python.csv', index=False)