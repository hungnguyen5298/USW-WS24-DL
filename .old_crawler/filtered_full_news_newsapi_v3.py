import pprint
import requests     # 2.19.1

secret = '923a0fe1e40f46d39ee8c9b9bc37fe8e'

# Define the endpoint
url = 'https://newsapi.org/v2/everything?'
# Specify the query and number of returns
parameters = {
    'q': 'APPL', # query phrase
    'pageSize': 20,  # maximum is 100
    'apiKey': secret # your own API key
}


# Make the request
response = requests.get(url, params=parameters)

# Convert the response to JSON format
response_json = response.json()

# Check out the dictionaries keys
print(response_json.keys())

pprint.pprint(response_json['articles'][0])