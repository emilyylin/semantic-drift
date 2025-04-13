import requests
from bs4 import BeautifulSoup

url = 'https://hughmcguire.medium.com/why-can-t-we-read-anymore-503c38c131fe'
headers = {'User-Agent': 'Mozilla/5.0'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# extract paragraph texts
paragraphs = soup.find_all('p')
content = '\n'.join([p.get_text(strip=True) for p in paragraphs])

print(content)
