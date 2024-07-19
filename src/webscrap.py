import scrapy
import json
from urllib.parse import urljoin, urlparse, urldefrag
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
import os

class MySpider(scrapy.Spider):
    name = 'MySpider'

    def __init__(self, start_url, depth=1, domain=None, allowed_file_extensions=(), *args, **kwargs):
        """Initializes the spider with the starting URL, domain, depth limit, and allowed file extensions.

        Args:
            start_url (str): The URL to start scraping from.
            depth (int, optional): The maximum depth of the crawl. Defaults to 1.
            domain (str, optional): The domain to allow for scraping. Defaults to None.
            allowed_file_extensions (tuple, optional): A tuple of allowed file extensions for download. Defaults to empty tuple.
        """
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        self.allowed_domains = [domain or start_url.split('://')[1].split('/')[0]]
        self.depth_limit = depth
        self.scraped_items = []
        self.page_count = 0
        self.max_pages = 1000
        self.visited_urls = set()
        self.non_text_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx')
        self.allowed_file_extensions = allowed_file_extensions

    def start_requests(self):
        """Starts the scraping process by making the initial request."""
        yield scrapy.Request(url=self.start_url, callback=self.parse, meta={'depth': 0})

    def parse(self, response):
        """Parses the response from the server and extracts links and text.

        Args:
            response (scrapy.http.Response): The response object from the request.
        """
        if self.page_count >= self.max_pages:
            self.crawler.engine.close_spider(self, reason='page limit reached')
            return

        content_type = response.headers.get('Content-Type', b'').decode('utf-8').lower()
        depth = response.meta['depth']

        if depth <= self.depth_limit:
            if 'text/html' in content_type:
                self.page_count += 1
                text_content = response.xpath('//text()[not(ancestor::style) and not(ancestor::script)]').getall()
                cleaned_text = ' '.join([t.strip() for t in text_content if t.strip()])
                extracted_links = self.extract_links_from_text(cleaned_text, response.url)

                for link in extracted_links:
                    sublink_url = urljoin(response.url, link)
                    sublink_url = urldefrag(sublink_url)[0]
                    if self.is_valid_url(sublink_url) and self.should_follow(sublink_url) and sublink_url not in self.visited_urls:
                        self.visited_urls.add(sublink_url)
                        self.logger.info(f"Constructed sublink from text: {sublink_url}")
                        yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

                item = {'url': response.url, 'content_type': content_type, 'text': cleaned_text}
                self.scraped_items.append(item)

            for link in response.xpath('//a/@href').extract():
                sublink_url = urljoin(response.url, link)
                sublink_url = urldefrag(sublink_url)[0]
                if self.is_valid_url(sublink_url) and self.should_follow(sublink_url) and sublink_url not in self.visited_urls:
                    self.visited_urls.add(sublink_url)
                    if self.is_downloadable_file(sublink_url):
                        self.logger.info(f"Downloading file: {sublink_url}")
                        yield scrapy.Request(url=sublink_url, callback=self.save_file)
                    else:
                        yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

    def closed(self, reason):
        """Handles the closing of the spider and saves scraped items to a JSON file.

        Args:
            reason (str): The reason for closing the spider.
        """
        with open('../data/scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.scraped_items, f, ensure_ascii=False, indent=4)

    def extract_links_from_text(self, text, base_url):
        """Extracts links embedded within text content.

        Args:
            text (str): The text content to extract links from.
            base_url (str): The base URL for resolving relative links.

        Returns:
            list: A list of extracted links.
        """
        selector = Selector(text=text)
        links = [href.strip() for href in selector.xpath('//a/@href').extract() if href]
        return links

    def is_valid_url(self, url):
        """Check if the URL is valid and starts with http or https.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        return url.startswith('http://') or url.startswith('https://')

    def should_follow(self, url):
        """Determine if the spider should follow the URL.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL should be followed, False otherwise.
        """
        return not self.is_non_text_url(url)

    def is_non_text_url(self, url):
        """Check if the URL is a non-text URL based on its file extension.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is a non-text URL, False otherwise.
        """
        parsed_url = urlparse(url)
        file_extension = f'.{parsed_url.path.split(".")[-1].lower()}'
        return file_extension in self.non_text_extensions

    def is_downloadable_file(self, url):
        """Check if the URL is for a downloadable file based on its file extension.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is for a downloadable file, False otherwise.
        """
        parsed_url = urlparse(url)
        file_extension = f'.{parsed_url.path.split(".")[-1].lower()}'
        return file_extension in self.allowed_file_extensions

    def save_file(self, response):
        """Save the downloaded file to the local filesystem.

        Args:
            response (scrapy.http.Response): The response object from the request.
        """
        path = urlparse(response.url).path
        filename = os.path.basename(path)
        with open(filename, 'wb') as file:
            file.write(response.body)
        self.logger.info(f"File saved: {filename}")

def main():
    start_url = 'https://docs.nvidia.com/cuda/'
    depth = 5
    allowed_file_extensions = ('.pdf', '.docx')

    process = CrawlerProcess(settings={
        'LOG_ENABLED': True,
        'CONCURRENT_REQUESTS': 32,
        'DOWNLOAD_DELAY': 1,
        'REACTOR_THREADPOOL_MAXSIZE': 20,
        'COOKIES_ENABLED': False,
        'RETRY_ENABLED': False,
        'REDIRECT_ENABLED': False,
    })

    process.crawl(MySpider, start_url=start_url, depth=depth, allowed_file_extensions=allowed_file_extensions)
    process.start()

if __name__ == '__main__':
    main()
