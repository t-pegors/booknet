#!/usr/bin/env python3
"""
GOOGLE BOOKS ENHANCER
=====================
Fetches metadata from Google Books API to populate the 'google' slot of Book objects.
"""

import os
import json
import logging
import time
import requests
import sys
from typing import Dict, Optional
from datetime import datetime

#### IMPORT SETUP ####

# Add the project root to the Python path so we can run this standalone or via main.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.goodreads_csv_extractor import Book, GoogleBooksData
from config import LIBRARY_PATH

# We still use os.getenv here because API keys belong in the .env file, not config.py!
#logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Google_Books_API")

#### HELPER FUNCTIONS ####

def setup_logger(timestamp: str):
    """Configures a logger that syncs with the experiment_runner timestamp."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("Google_Books_API")
    logger.setLevel(logging.DEBUG) # Capture everything
    
    # Prevent duplicate logs if run multiple times in same session
    if not logger.handlers: 
        fh = logging.FileHandler(os.path.join(log_dir, f"googlebooks_ingest_{timestamp}.log"), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Keep console clean, only show INFO
        
        formatter = logging.Formatter('%(asctime)s | [%(levelname)s] | %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

def get_google_api_key() -> str:
    """Retrieves API Key from env or throws error."""
    key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not key:
        logger.error("CRITICAL: GOOGLE_BOOKS_API_KEY is missing from .env file.")
        sys.exit(1)
    return key

def make_api_request(query: str, api_key: str) -> Optional[Dict]:
    """
    Sends a request to Google Books API.
    Includes rate limiting.
    """
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "key": api_key,
        "maxResults": 1  # We only want the best match
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        time.sleep(0.2) # Rate limit: 5 calls/sec max
        
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                return data["items"][0]  # Return the full JSON object
            return None
        elif response.status_code == 429:
            logger.warning("Rate limit hit. Sleeping 5s...")
            time.sleep(5)
            return None
        else:
            logger.warning(f"API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

def extract_google_data(api_result: Dict) -> GoogleBooksData:
    """
    Parses the raw JSON into our Dataclass.
    Saves the ENTIRE json blob into 'raw_data' for safekeeping.
    """
    info = api_result.get("volumeInfo", {})
    
    return GoogleBooksData(
        id=api_result.get("id", ""),
        title=info.get("title", ""),
        subtitle=info.get("subtitle", ""),
        authors=info.get("authors", []),
        description=info.get("description", ""),
        categories=info.get("categories", []),
        published_date=info.get("publishedDate", ""),
        thumbnail_url=info.get("imageLinks", {}).get("thumbnail", ""),
        page_count=info.get("pageCount", 0),
        language=info.get("language", "en"),
        
        # Save the entire API result here so we never lose data.
        raw_data=api_result
    )

def fetch_metadata_for_book(book: Book, api_key: str, force_update: bool = False):
    """
    Orchestrates the fetch logic for a single book.
    """
    # Skip if we already have an ID (unless forced)
    if book.google.id and not force_update:
        return 
    
    logger.info(f"Fetching: {book.goodreads.title}")
    result = None

    # STRATEGY 1: ISBN (Best)
    isbn = book.goodreads.isbn13 or book.goodreads.isbn
    if isbn:
        result = make_api_request(f"isbn:{isbn}", api_key)
    
    # STRATEGY 2: Title + Author (Fallback)
    if not result:
        # cleanup title slightly for better search (remove subtitles after colon)
        clean_title = book.goodreads.title.split(':')[0]
        query = f"intitle:{clean_title} inauthor:{book.goodreads.author}"
        result = make_api_request(query, api_key)

    if result:
        book.google = extract_google_data(result)
        logger.info(f"  -> Match: {book.google.title}")
    else:
        logger.warning(f"  -> No match found.")

def process_books(library: list[Book], api_key: str, force_update: bool = False, progress_callback=None):
    """
    Processes a list of books in-memory.
    Completely decoupled from the hard drive.
    """
    needs_update = [b for b in library if not b.google.id or force_update]
    logger.info(f"Books to process: {len(needs_update)} / {len(library)}")

    for i, book in enumerate(library):
        if not book.google.id or force_update:
            # 1. Update the book in memory
            fetch_metadata_for_book(book, api_key, force_update=force_update)
            
            # 2. Tell the caller (CLI or Streamlit) that a step finished
            if progress_callback:
                progress_callback(i, len(library), book)

#### MAIN EXECUTION ####

def main():
    """CLI WRAPPER: Handles disk I/O and runs the process."""
    # Initialize Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(timestamp)

    logger.info(f"--- Starting Google Books API Session [{timestamp}] ---")

    API_KEY = get_google_api_key()
    FORCE_UPDATE = os.getenv("GOOGLE_FORCE_UPDATE", "False").lower() == "true"

    if not os.path.exists(LIBRARY_PATH):
        logger.error(f"CRITICAL: Input file not found at {LIBRARY_PATH}")
        return

    try:
        # 1. Load the JSON from the hard drive
        logger.info(f"Loading library from {LIBRARY_PATH}...")
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        library = [Book.from_dict(item) for item in data]
        
        # 2. Define what the CLI should do after every book is processed
        def cli_save_progress(index, total, current_book):
            # Progressive Save: Write to disk every 10 updates
            if index > 0 and index % 10 == 0:
                with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
                    json.dump([b.to_dict() for b in library], f, indent=4)
                    
        # 3. Pass the library and the callback into the core logic
        process_books(library, API_KEY, force_update=FORCE_UPDATE, progress_callback=cli_save_progress)

        # 4. Final Save to disk
        with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in library], f, indent=4)
        
        logger.info("Success! Google Books metadata updated.")

    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()