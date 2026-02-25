#!/usr/bin/env python3
"""
PROJECT ORCHESTRATOR
====================
This is the main entry point for the project. 
It triggers the individual modules in the correct order.
"""

import logging
import sys

# Import your module
try:
    from pipeline.goodreads_csv_extractor import main as run_goodreads
    from pipeline.googlebooks_enhancer import main as run_google_fetcher
    from pipeline.goodreads_scraper import main as run_goodreads_scraper
    from pipeline.ai_enrichment import main as run_ai_enrichment
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules. Is the 'pipeline' folder missing an __init__.py? \nError: {e}")
    sys.exit(1)

# Setup central logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Orchestrator")

def run_pipeline():
    """
    Runs the data pipeline steps in order.
    """
    logger.info("=== STARTING DATA PIPELINE ===")

    try:
        # STEP 1: Goodreads Extraction
        logger.info("Step 1: Running Goodreads Extractor...")
        run_goodreads()
        logger.info("Step 1: Complete.")

        # STEP 2: Google Books API
        logger.info("Step 2: Fetching Google Metadata...")
        run_google_fetcher()
        logger.info("Step 2: Complete.")

        # STEP 3: Goodreads Scraper
        logger.info("Step 3: Fetching Goodreads descriptions...")
        run_goodreads_scraper()
        logger.info("Step 3: Complete.")

        # STEP 4: LLM Enrichment
        logger.info("Step 4: Running LLM Analysis...")
        run_ai_enrichment()
        logger.info("Step 4: Complete")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        sys.exit(1)
    
    logger.info("=== PIPELINE FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_pipeline()