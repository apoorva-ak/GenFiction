import scrapy
from scrapy.crawler import CrawlerProcess
from twisted.internet import asyncioreactor
asyncioreactor.install()
from twisted.internet import reactor
from multiprocessing.context import Process
import pandas as pd

class FlashFictionSpider(scrapy.Spider):
    name = 'flash_fiction'
    start_urls = ['https://everydayfiction.com/ink-by-sarthak-sharma/']

    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'everyday_fiction_data.csv',
    }

    def parse(self, response):
        title_author = response.css('.entry-title::text').get().strip()
        title, author = map(str.strip, title_author.split('•', 1))

        # Extract article content
        content = ' '.join(response.css('.entry-content p::text').getall())

        yield {
            'url': response.url,
            'title': title,
            'author': author,
            'content': content
        }

        next_article_url = response.css('a[rel="prev"]::attr(href)').get()
        if next_article_url:
            yield scrapy.Request(next_article_url, callback=self.parse)


def crawl():
    crawler = CrawlerProcess()
    crawler.crawl(FlashFictionSpider)
    crawler.start()

process = Process(target=crawl)
process.start()
process.join()
