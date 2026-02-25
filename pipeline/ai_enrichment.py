import requests
import json
import os
import time 
import logging
import re
import sys
from typing import Any
from datetime import datetime

from pipeline.goodreads_csv_extractor import Book
from pipeline.category_loader import get_genres, get_spheres, get_spheres_with_descriptions, get_eras, get_era_definitions, get_all_location_leaves

#######################
#### CONFIGURATION ####
#######################

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from central config
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, ADD_REASONING_OUTPUT, TEMPERATURE, IS_REASONING_MODEL, ADD_STEP_TIMING, LIBRARY_PATH

logger = logging.getLogger("AI_Enrichment")

def setup_logger(timestamp: str):
    """Configures a logger that syncs with the experiment_runner timestamp."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("AI_Enrichment")
    logger.setLevel(logging.DEBUG) # Capture everything
    
    # Prevent duplicate logs if run multiple times in same session
    if not logger.handlers: 
        fh = logging.FileHandler(os.path.join(log_dir, f"ai_enrichment_{timestamp}.log"), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Keep console clean, only show INFO
        
        formatter = logging.Formatter('%(asctime)s | [%(levelname)s] | %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

def build_prompt(context: str, task: str, reasoning: bool, is_reasoning_model: bool, output_options: str) -> str:
    """Builds a prompt for the LLM."""

    prompt = f"""
              Context: {context}
              Task: {task}
             """

    if is_reasoning_model:
        # reasoning models need the freedom to think in a structured way
        prompt += "\nPlease think step-by-step. Wrap your thoughts in <think> and </think> tags.\n"
        prompt += 'After thinking, return ONLY a valid JSON object with a single key named "answer".\n'
    elif reasoning:
        # non-reasoning models need to be guided to provide reasoning
        prompt += "\nPlease provide your reasoning before giving the final answer.\n"
        prompt += "Wrap your thought process strictly inside <scratchpad> and </scratchpad> tags.\n"
        prompt += 'After the scratchpad, return ONLY a valid JSON object with a single key named "answer".\n'
    else:
        prompt += '\nRreturn ONLY a valid JSON object with a single key named "answer".\n'

    if output_options:
        prompt += f"\n{output_options}"

    return prompt

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Helper to send requests to local Ollama instance."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m", # keeps the model loaded in VRAM between calls
        "options": {
            "temperature": float(TEMPERATURE),
            "num_predict": 1200, # Hard caps output so it doesn't ramble forever
            "num_ctx": 8192 # add a large context window to handle long prompts
        }
    }

    if not IS_REASONING_MODEL:
        payload["format"] = "json"

    logger.debug(f"\n>>> [{model}] SENDING PROMPT >>>\n{prompt}\n===========================\n")

    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload, timeout=90)
        if response.status_code != 200:
            logger.error(f"[{model}] Ollama API Error {response.status_code}: {response.text}")
            return "{}"
            
        raw_output = response.json().get('response', "{}")
        
        logger.debug(f"\n--- [{model}] RAW OUTPUT ---\n{raw_output}\n---------------------------\n")
        return raw_output
        
    except Exception as e:
        logger.error(f"[{model}] Request failed: {str(e)}")
        return "{}"

def clean_and_parse_json(raw_text: str) -> dict:
    """Hunts for JSON, extracts XML scratchpads/think tags, and logs them."""
    logger = logging.getLogger("AI_Enrichment")
    if not raw_text: return {}
    
    # Extract Reasoning from XML tags
    reasoning = ""
    scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', raw_text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL) # Catch Phi-4/DeepSeek native tags
    
    if scratchpad_match:
        reasoning = scratchpad_match.group(1).strip()
    elif think_match:
        reasoning = think_match.group(1).strip()
        
    # Log the extracted reasoning so you can read it in the log file
    if reasoning:
        logger.debug(f"\n🧠 EXTRACTED REASONING:\n{reasoning}\n" + "="*40)
        
    # Strip tags to isolate JSON
    raw_text = re.sub(r'<scratchpad>.*?</scratchpad>', '', raw_text, flags=re.DOTALL)
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    
    # Find JSON block
    match = re.search(r'```(?:json)?(.*?)```', raw_text, re.DOTALL)
    if match:
        clean_text = match.group(1).strip()
    else:
        raw_text = raw_text.strip()
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_text = raw_text[start_idx:end_idx + 1]
        else:
            logger.warning(f"No JSON brackets found in output. Raw text: {raw_text[:100]}...")
            return {}
            
    # Parse & Inject Reasoning back into the Python dictionary
    try:
        parsed = json.loads(clean_text)
        if reasoning: parsed["_scratchpad_reasoning"] = reasoning # Keeps HTML report working!
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"JSON Parse Error: {e}. Attempting recovery. Raw JSON was: {clean_text}")
        try:
            parsed = json.loads(clean_text.replace("'", '"'))
            if reasoning: parsed["_scratchpad_reasoning"] = reasoning
            return parsed
        except Exception:
            logger.error("JSON Recovery Failed.")
            return {}

def ensure_string(val: Any) -> str:
    #If val is a list, returns first element. Otherwise returns string representation.
    if isinstance(val, list):
        return str(val[0]) if len(val) > 0 else "Unknown"
    return str(val) if val is not None else "Unknown"

def enrich_book(book: Book, model: str = OLLAMA_MODEL) -> dict:

    """
    Runs a Chain-of-Thought pipeline.
    Step 1: Fiction / Non-fiction
    Step 2: Genre
    Step 3: Sphere (i.e. major world region)
    Step 4: Era (eras are unique to each sphere)
    Step 5: Location(s) (more specific locations, can be multiple)
    """
    
    metrics = {}
    
    book_context = f"""
                    Title from Goodreads: {book.goodreads.title} 
                    Original Year Published: {book.goodreads.original_pub_year}
                    Author: {book.goodreads.author}
                    Additional Authors: {book.goodreads.additional_authors}
                    Title from Google: {book.google.title}
                    Subtitle from Google: {book.google.subtitle}
                    Description: {book.google.description}
                    User Tags: {book.goodreads.bookshelves}
                    Categories: {book.google.categories}
                    """

    #### STEP 1: FICTION VS NON-FICTION ####

    if ADD_STEP_TIMING:
        t_start = time.time()

    prompt = build_prompt(context=book_context, 
                          task="Classify this book as 'Fiction' or 'Non-fiction' based on the above context",
                          reasoning=ADD_REASONING_OUTPUT, 
                          is_reasoning_model=IS_REASONING_MODEL,
                          output_options=f" The answer must be Fiction or Non-fiction.")

    r1 = clean_and_parse_json(query_ollama(prompt, model))
    main_category = ensure_string(r1.get("answer", "Unknown"))

    if ADD_REASONING_OUTPUT:
        cat_reasoning = r1.get("_scratchpad_reasoning", "")

    if ADD_STEP_TIMING:
        metrics['step_1_cat'] = round(time.time() - t_start, 2) # Capture time

    #### STEP 2: GENRE SELECTION ####

    if ADD_STEP_TIMING:
        t_start = time.time()

    available_genres = get_genres(main_category)

    prompt_2 = build_prompt(context=f"{main_category} book. {book_context}", 
                            task=f"""
                                  Select genre from: {json.dumps(available_genres)}. 
                                  Note: If this is a fiction book and there are clear historic references, choose Historical Fiction.
                                  """,
                            reasoning=ADD_REASONING_OUTPUT,
                            is_reasoning_model=IS_REASONING_MODEL,
                            output_options=f" The answer must be one of: {json.dumps(available_genres)}")

    r2 = clean_and_parse_json(query_ollama(prompt_2, model))
    genre = ensure_string(r2.get("answer", "Unknown"))

    if ADD_REASONING_OUTPUT:
        genre_reasoning = r2.get("_scratchpad_reasoning", "")
    if ADD_STEP_TIMING:
        metrics['step_2_genre'] = round(time.time() - t_start, 2)
   
    #### STEP 3: SPHERE SELECTION ####

    if ADD_STEP_TIMING:
        t_start = time.time()
    
    # Get the list of just names (for validation)
    valid_sphere_names = get_spheres()
    # Get the rich text descriptions (for the prompt)
    sphere_definitions = get_spheres_with_descriptions() 

    prompt_3 = build_prompt(context=f"{main_category} -> {genre}. {book_context}", 
                            task=f"""
                                   Assign this book to the most appropriate Sphere based ONLY on the definitions below.
                                   Definitions of Spheres:
                                    {sphere_definitions}
                                  Tie-Breaker Rules:
                                    - If a book is about theoretical science, mathematics, or abstract philosophy that is not tied to a specific geographic region, assign it to 'The Cosmos & Otherworlds'.
                                    - If the book features aliens, magic, or future tech, usually assign it to 'Cosmos & Otherworlds'—even if it starts in a real country (e.g., 'Three-Body Problem' -> Cosmos, not East Asia).
                                    - If a character travels between worlds (e.g., Iran to Europe), assign it to their 'Home' culture or where the primary historical conflict is rooted (e.g., 'Persepolis' -> Western World).
                                    - If a book is about a Western Empire invading another region, usually assign it to the region being invaded if the focus is on the local culture (e.g., 'Things Fall Apart' -> Global South).
                                  """,
                            reasoning=ADD_REASONING_OUTPUT,
                            is_reasoning_model=IS_REASONING_MODEL,
                            output_options=f" The answer must be one of: {json.dumps(valid_sphere_names)}")

    r3 = clean_and_parse_json(query_ollama(prompt_3, model))
    sphere = ensure_string(r3.get("answer", "Unknown"))

    if ADD_REASONING_OUTPUT:
        sphere_reasoning = r3.get("_scratchpad_reasoning", "")

    # Validation / Fallback
    if sphere not in valid_sphere_names:
        sphere = f"⚠️ Invalid sphere: '{sphere}'."

    if ADD_STEP_TIMING:        
        metrics['step_3_sphere'] = round(time.time() - t_start, 2)

    #### STEP 4: ERA SELECTION ####
    if ADD_STEP_TIMING:
        t_start = time.time()
    
    # Get the list of just names (for validation)
    valid_era_names = get_eras(sphere)
    # Get the rich text descriptions (for the prompt)
    era_definitions = get_era_definitions(sphere) 
    
    prompt_4 = build_prompt(context=f"{main_category} -> {genre} -> {sphere}. {book_context}", 
                            task=f"""
                                   Select the overarching Era.
                                   Era Definitions (Use these detailed periods to guide your choice):
                                   {era_definitions}
                                   Use the following steps to determine the best era:
                                    - Identify specific time periods or events in the book (e.g. 'World War II', 'Space Travel', 'The 90s').
                                    - Match them to the 'Includes...' Era Definitions above.
                                    - Select the overarching Era.
                                    """,
                            reasoning=ADD_REASONING_OUTPUT,
                            is_reasoning_model=IS_REASONING_MODEL,
                            output_options=f" The answer must be one of: {json.dumps(list(valid_era_names.keys()))}")
    
    r4 = clean_and_parse_json(query_ollama(prompt_4, model))
    era = ensure_string(r4.get("answer", "Unknown"))
    if ADD_REASONING_OUTPUT:
        era_reasoning = r4.get("_scratchpad_reasoning", "")

    # Validation / Fallback
    if era not in valid_era_names:
        era = f"⚠️ Invalid era: '{era}'."

    if ADD_STEP_TIMING:
        metrics['step_4_era'] = round(time.time() - t_start, 2)

    #### STEP 5: LOCATION SELECTION ####
    if ADD_STEP_TIMING:
        t_start = time.time()
    
    # Get the flat list of specific cities/sites
    valid_loc_names = get_all_location_leaves()
    
    prompt_5 = build_prompt(context=f"{main_category} -> {genre} -> {sphere} -> {era}. {book_context}", 
                            task=f"""
                                   Identify the specific primary setting(s) of the book.

                                   Approved Location List:
                                   {json.dumps(valid_loc_names)}

                                   Use the following steps to determine the best settings / locations:                                 
                                    - Identify the primary setting(s) of the book.
                                    - From the list above, select ONE or MORE of the locations mentioned above.                                   
                                    - You can select MULTIPLE locations if the story travels or has multiple settings.
                                    - You CAN select locations outside the '{sphere}' if the story takes place there (e.g. A Western biography set in Japan).
                                   
                                    (You locations must be selected from the approved list above.)
                                   """,
                            reasoning=ADD_REASONING_OUTPUT,
                            is_reasoning_model=IS_REASONING_MODEL,
                            output_options=f"The answer MUST be a JSON list of strings.")
    
    r5 = clean_and_parse_json(query_ollama(prompt_5, model))
    locations = r5.get("answer") or r5.get("answers") or []
    
    # Ensure it's a list
    if isinstance(locations, str):
        locations = [locations]
    elif isinstance(locations, dict):
        locations = list(locations.values())

    cleaned_locations = []
    for loc in locations:
        # If response nested a dict inside the list e.g. [{"country": "China"}]
        if isinstance(loc, dict):
            loc = next(iter(loc.values())) if loc else ""
            
        loc_str = str(loc)
        if "," in loc_str:
            for l in loc_str.split(","):
                clean_l = l.strip(" []'\"") 
                if clean_l: cleaned_locations.append(clean_l)
        else:
            # Properly escaped double quote
            clean_str = loc_str.strip(" []'\"")
            if clean_str: cleaned_locations.append(clean_str)
            
    locations = cleaned_locations
    
    # Validation Step: Only keep tags that exactly match your approved list
    locations = [loc for loc in locations if loc in valid_loc_names]
    
    if ADD_REASONING_OUTPUT:
        loc_reasoning = r5.get("_scratchpad_reasoning", "")

    if ADD_STEP_TIMING:
        metrics['step_5_loc'] = round(time.time() - t_start, 2)
        metrics['total_time'] = round(sum(metrics.values()), 2)

    if ADD_REASONING_OUTPUT:
        return {
            "model_used": model,
            "category": main_category,
            "cat_reasoning": cat_reasoning,
            "genre": genre,
            "genre_reasoning": genre_reasoning,
            "sphere": sphere,
            "sphere_reasoning": sphere_reasoning,
            "era": era,
            "era_reasoning": era_reasoning,
            "location": locations, 
            "loc_reasoning": loc_reasoning,
            "metrics": metrics
        }
    else:
        return {
            "model_used": model,
            "category": main_category,
            "genre": genre,
            "sphere": sphere,
            "era": era,
            "location": locations, 
            "metrics": metrics
        }

def process_books(library: list[Book], model: str = OLLAMA_MODEL, force_update: bool = False, progress_callback=None):
    """
    Processes the library in-memory, mapping AI results to the Book objects.
    """
    # Find books that don't have a category yet, or force update
    needs_update = [b for b in library if not b.llm.category or force_update]
    logger.info(f"Books to enrich: {len(needs_update)} / {len(library)}")

    for i, book in enumerate(library):
        if not book.llm.category or force_update:
            # 1. Run the AI pipeline
            ai_result = enrich_book(book, model=model)
            
            # 2. Map the results to the dataclass
            book.llm.category = ai_result.get("category", "")
            book.llm.genre = ai_result.get("genre", "")
            book.llm.sphere = ai_result.get("sphere", "")
            book.llm.era = ai_result.get("era", "")
            book.llm.location = ai_result.get("location", [])

            # Take a permanent snapshot of the AI's exact answers
            book.llm.ai_snapshot = {
                "category": book.llm.category,
                "genre": book.llm.genre,
                "sphere": book.llm.sphere,
                "era": book.llm.era,
                "location": book.llm.location
            }
            
            if ADD_REASONING_OUTPUT:
                book.llm._reasoning_log = {
                    "category": ai_result.get("cat_reasoning", ""),
                    "genre": ai_result.get("genre_reasoning", ""),
                    "sphere": ai_result.get("sphere_reasoning", ""),
                    "era": ai_result.get("era_reasoning", ""),
                    "location": ai_result.get("loc_reasoning", "")
                }

            # 3. Trigger the callback
            if progress_callback:
                progress_callback(i, len(library), book)

def main():
    """CLI WRAPPER: Handles disk I/O and runs the enrichment process."""

    # Initialize Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(timestamp)
    
    logger.info(f"--- Starting AI Enrichment Session [{timestamp}] ---")

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
            logger.info(f"✅ Enriched: {current_book.goodreads.title} -> {current_book.llm.category} / {current_book.llm.genre}")
            # Progressive Save: Write to disk every 5 updates (AI is slow, save frequently!)
            if index > 0 and index % 5 == 0:
                with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
                    json.dump([b.to_dict() for b in library], f, indent=4)
                    
        # 3. Pass the library into the core logic
        process_books(library, model=OLLAMA_MODEL, force_update=False, progress_callback=cli_save_progress)

        # 4. Final Save
        with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in library], f, indent=4)
        
        logger.info("Success! AI Enrichment complete.")

    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()