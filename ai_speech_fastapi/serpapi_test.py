# import requests
# from serpapi import GoogleSearch

# api_key = "6841e4241b995fcef2024bbb"

# def fetch_data_from_serpapi(query):
#     search_params = {
#         "q": query,
#         "api_key": api_key,
#         "engine": "google"
#     }

   
#     search = GoogleSearch(search_params)
#     results = search.get_dict()

#     if 'organic_results' in results:
#         return results['organic_results']
#     else:
#         return "No results found."

# topic = "Python"
# results = fetch_data_from_serpapi(topic)

# for idx, result in enumerate(results, 1):
#     print(f"{idx}. Title: {result.get('title')}")
#     print(f"   Link: {result.get('link')}")
#     print(f"   Snippet: {result.get('snippet')}\n")


import requests

api_key = "6875f0032c08c57c92e85ffd"

from bs4 import BeautifulSoup


url = "https://en.wikipedia.org/wiki/Association_football"

# Step 1: Get HTML from Scrapingdog
api_endpoint = f"https://api.scrapingdog.com/scrape?api_key={api_key}&url={url}"
response = requests.get(api_endpoint)

if response.status_code == 200:
    # Step 2: Use BeautifulSoup to parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Optional: Remove script/style tags for cleaner content
    for script in soup(["script", "style", "noscript"]):
        script.decompose()

    # Step 3: Extract visible text only
    text = soup.get_text(separator="\n", strip=True)

    # Print or use the clean text
    print(text)
else:
    print(f"Error: {response.status_code} - {response.text}")


