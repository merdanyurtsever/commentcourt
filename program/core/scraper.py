from bs4 import BeautifulSoup
import requests
from program.utils.config import load_config
from program.utils.io import save_json

class BaseScraper:
    def __init__(self, name, storefronts):
        self.name = name
        self.storefronts = storefronts
        self.data = []

    def scrape(self):
        for url in self.storefronts:
            if "trendyol" in url:
                self.data.extend(self.scrape_trendyol(url))
            elif "hepsiburada" in url:
                self.data.extend(self.scrape_hepsiburada(url))
            elif "n11" in url:
                self.data.extend(self.scrape_n11(url))
            elif "milla" in url:
                self.data.extend(self.scrape_milla(url))
        return self.data

    def scrape_trendyol(self, url):
        # TODO: Trendyol yorum scraping
        return []

    def scrape_hepsiburada(self, url):
        # TODO: Hepsiburada yorum scraping
        return []

    def scrape_n11(self, url):
        # TODO: N11 yorum scraping
        return []

    def scrape_milla(self, url):
        # TODO: Milla yorum scraping
        return []

def run_all_scrapers():
    config = load_config()
    all_data = {}
    for influencer in config["influencers"]:
        scraper = BaseScraper(influencer["name"], influencer["storefronts"])
        data = scraper.scrape()
        all_data[influencer["name"]] = data
        save_json(f'database/raw/{influencer["name"]}.json', data)
    return all_data
