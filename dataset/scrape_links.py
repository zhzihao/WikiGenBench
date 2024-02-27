import os
import json
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import argparse

headers = [
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0 cb) like Gecko",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko)",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36",
    },
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    },
]

proxies = [
    {"http": "103.154.231.133:8080"},
    {"http": "111.160.169.54:41820"},
    {"http": "188.132.222.37:8080"},
    {"http": "103.65.233.38:80"},
    {"http": "110.49.11.50:8080"},
    {"http": "45.228.234.92:999"},
    {"http": "50.206.25.107:80"},
    {"http": "116.50.163.67:80"},
    {"http": "51.159.66.158:3128"},
    {"http": "138.94.236.71:8080"},
    {"http": "167.249.29.218:999"},
    {"http": "140.227.198.33:3128"},
    {"http": "34.66.5.144:8888"},
    {"http": "106.249.44.10:3128"},
    {"http": "103.73.74.217:2021"},
    {"http": "103.106.193.117:7532"},
    {"http": "103.152.232.124:3125"},
    {"http": "45.8.104.9:80"},
    {"http": "180.94.64.58:8080"},
    {"http": "36.95.54.114:8080"},
    {"http": "196.3.97.71:23500"},
    {"http": "104.239.113.74:3199"},
    {"http": "91.194.239.122:8080"},
    {"http": "104.26.4.192:80"},
    {"http": "36.92.111.49:9812"},
    {"http": "202.180.19.41:8080"},
    {"http": "162.159.244.202:80"},
    {"http": "12.186.206.83:80"},
    {"http": "194.111.185.23:80"},
    {"http": "37.228.65.107:32052"},
    {"http": "37.77.134.218:80"},
    {"http": "193.169.189.25:80"},
    {"http": "38.41.29.78:8080"},
    {"http": "153.126.200.201:80"},
    {"http": "104.248.227.150:80"},
    {"http": "120.89.90.230:80"},
    {"http": "46.101.172.5:80"},
    {"http": "190.186.1.65:999"},
    {"http": "185.189.186.19:8080"},
    {"http": "5.44.54.16:8080"},
    {"http": "51.91.111.43:80"},
    {"http": "43.228.95.122:83"},
    {"http": "110.235.249.226:8080"},
    {"http": "47.109.51.138:80"},
    {"http": "139.99.236.128:3128"},
    {"http": "45.174.70.18:53281"},
    {"http": "165.227.36.36:80"},
    {"http": "165.227.188.89:80"},
    {"http": "158.51.107.253:8080"},
    {"http": "138.118.50.253:8080"},
    {"http": "170.246.85.108:50991"},
    {"http": "162.240.75.37:80"},
    {"http": "45.173.6.5:999"},
    {"http": "182.253.197.60:80"},
    {"http": "103.156.141.100:80"},
    {"http": "144.48.60.104:12444"},
    {"http": "146.56.144.24:5000"},
    {"http": "85.238.104.216:8088"},
    {"http": "139.162.182.54:4916"},
    {"http": "117.54.114.100:80"},
    {"http": "165.22.3.209:3128"},
    {"http": "104.18.2.10:80"},
    {"http": "154.64.117.226:80"},
    {"http": "185.195.106.231:23445"},
    {"http": "36.94.48.188:2021"},
    {"http": "113.160.235.248:19132"},
    {"http": "79.127.56.147:8080"},
    {"http": "190.185.116.161:999"},
    {"http": "104.129.192.32:8800"},
    {"http": "178.212.196.177:9999"},
    {"http": "103.175.46.8:3125"},
    {"http": "103.122.90.254:80"},
    {"http": "185.169.232.233:12000"},
    {"http": "154.113.19.30:8080"},
    {"http": "172.67.72.1:80"},
    {"http": "185.121.26.132:24399"},
    {"http": "123.182.59.57:8089"},
    {"http": "190.152.8.74:41890"},
    {"http": "194.44.93.102:3128"},
    {"http": "45.156.29.57:9090"},
    {"http": "142.93.96.177:80"},
    {"http": "36.94.30.126:8080"},
    {"http": "196.203.83.249:9090"},
    {"http": "103.15.140.121:44759"},
    {"http": "173.249.30.165:3128"},
    {"http": "190.12.95.170:47029"},
    {"http": "185.198.190.164:12444"},
    {"http": "88.150.230.197:80"},
    {"http": "185.195.107.13:23445"},
    {"http": "39.108.230.16:3128"},
    {"http": "139.130.87.162:8080"},
    {"http": "45.156.29.59:9090"},
    {"http": "174.81.78.64:48678"},
    {"http": "188.133.173.21:8080"},
    {"http": "175.45.195.18:80"},
    {"http": "62.3.30.22:8080"},
    {"http": "112.87.140.164:9401"},
    {"http": "185.209.229.26:80"},
    {"http": "94.181.48.110:1256"},
    {"http": "203.32.120.218:80"},
    {"http": "164.52.206.180:80"},
]


def scrape_url(url):
    try:
        response = requests.get(url, headers=random.choice(headers), proxies=random.choice(proxies), timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return url, soup.get_text()
    except requests.RequestException as e:
        print(f"Error scraping URL {url}: {e}")
        return url, None

def process_file(file_path, output_directory):
    file_id = os.path.basename(file_path).split('.')[0]
    print(f'Scraping for file ID: {file_id}')

    with open(file_path, 'r', encoding='utf-8') as file:
        links_data = json.load(file)

    scraped_data = {}
    for link_data in links_data:
        url = link_data['link']
        scraped_data[url] = scrape_url(url)[1]  # Only store the scraped text

    output_filepath = os.path.join(output_directory, f'{file_id}.json')
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(scraped_data, file, indent=4)
    print(f'Completed scraping for file ID: {file_id}')

def process_links(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    json_files = [f for f in os.listdir(input_directory) if f.endswith('.json')]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, os.path.join(input_directory, file), output_directory) for file in json_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            pass

def main():
    parser = argparse.ArgumentParser(description="Scrape links from JSON files and store the data.")
    parser.add_argument("--input_directory", type=str, default='output/search_link', help="Directory containing input JSON files with links.")
    parser.add_argument("--output_directory", type=str, default='output/scraped_data', help="Directory to store the scraped data.")
    args = parser.parse_args()

    process_links(args.input_directory, args.output_directory)

if __name__ == "__main__":
    main()
