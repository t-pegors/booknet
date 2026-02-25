#!/usr/bin/env python3
"""
GOODREADS SCRAPER
=================
Stealthily scrapes missing book descriptions directly from Goodreads HTML.
"""
import os
import sys
import json
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime

#### IMPORT SETUP ####
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.goodreads_csv_extractor import Book
from config import LIBRARY_PATH

logger = logging.getLogger("Goodreads_Scraper")

#### CONFIGURATION ###

def setup_logger(timestamp: str):
    """Configures a logger that syncs with the experiment_runner timestamp."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("Goodreads_Scraper")
    logger.setLevel(logging.DEBUG) # Capture everything
    
    # Prevent duplicate logs if run multiple times in same session
    if not logger.handlers: 
        fh = logging.FileHandler(os.path.join(log_dir, f"goodreads_scraper_{timestamp}.log"), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Keep console clean, only show INFO
        
        formatter = logging.Formatter('%(asctime)s | [%(levelname)s] | %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

# Configure Python script as a web browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

def fetch_description_from_html(book_id: str) -> str:
    """Hits the Goodreads URL and parses the HTML for the description."""
    url = f"https://www.goodreads.com/book/show/{book_id}"
    
    try:
        # 1. Insert random delay (2.5 to 4.5 seconds)
        time.sleep(random.uniform(2.5, 4.5))
        
        # 2. Make the request
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code == 403:
            logger.error("🚨 403 Forbidden! Goodreads blocked the request. Pausing for 15 seconds...")
            time.sleep(15)
            return ""
            
        if response.status_code != 200:
            logger.warning(f"Failed to load page for ID {book_id}. Status: {response.status_code}")
            return ""
            
        # 3. Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 4. Hunt for the description (Goodreads uses a few different layouts)
        description = ""
        
        # Target A: The modern layout
        desc_div = soup.find('div', {'data-testid': 'description'})
        if desc_div:
            description = desc_div.get_text(separator='\n', strip=True)
            
        # Target B: Older layout fallback
        if not description:
            desc_div = soup.find('div', id='description')
            if desc_div:
                spans = desc_div.find_all('span')
                # Usually the second span has the full text if there's a "Read more" toggle
                if len(spans) > 1:
                    description = spans[1].get_text(separator='\n', strip=True)
                elif spans:
                    description = spans[0].get_text(separator='\n', strip=True)

        return description
        
    except Exception as e:
        logger.error(f"Error scraping ID {book_id}: {e}")
        return ""

def process_books(library: list[Book], force_update: bool = False, progress_callback=None):
    """
    Scrapes Goodreads for descriptions.
    Only targets books that have NO description from Google AND NO description from Goodreads.
    """
    needs_update = [
        b for b in library 
        if (not getattr(b.goodreads, 'description', None) and not b.google.description) or force_update
    ]
    
    logger.info(f"Books needing Goodreads scrape: {len(needs_update)} / {len(library)}")

    for i, book in enumerate(library):
        # Check both places so we don't waste time scraping data we already have
        has_gr_desc = getattr(book.goodreads, 'description', "")
        has_google_desc = book.google.description
        
        if (not has_gr_desc and not has_google_desc) or force_update:
            desc = fetch_description_from_html(book.goodreads.id)
            
            if desc:
                book.goodreads.description = desc
            
            if progress_callback:
                progress_callback(i, len(library), book)

def main():
    """CLI WRAPPER: Handles disk I/O and runs the scraper."""
    
    # Initialize Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(timestamp)
    
    logger.info("--- Starting Goodreads Scraping ---")
    
    if not os.path.exists(LIBRARY_PATH):
        logger.error(f"CRITICAL: Input file not found at {LIBRARY_PATH}")
        return

    try:
        logger.info(f"Loading library from {LIBRARY_PATH}...")
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        library = [Book.from_dict(item) for item in data]
        
        def cli_save_progress(index, total, current_book):
            if getattr(current_book.goodreads, 'description', ""):
                logger.info(f"✅ Scraped: {current_book.goodreads.title}")
            else:
                logger.warning(f"⏭️ No description found for: {current_book.goodreads.title}")
                
            # Progressive Save: Write to disk every 5 updates (Scraping is slow!)
            if index > 0 and index % 5 == 0:
                with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
                    json.dump([b.to_dict() for b in library], f, indent=4)
                    
        process_books(library, force_update=False, progress_callback=cli_save_progress)

        with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in library], f, indent=4)
        
        logger.info("Success! Goodreads scraping complete.")

    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()