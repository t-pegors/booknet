#!/usr/bin/env python3
import random
import os
import json
import pandas as pd
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
from pipeline.goodreads_csv_extractor import Book
from pipeline.ai_enrichment import setup_logger, enrich_book


# --- CONFIG ---
from config import (
    LIBRARY_PATH, 
    GOLDEN_PATH, 
    EVAL_DIR,
    ADD_REASONING_OUTPUT, 
    ADD_STEP_TIMING,
    MODELS_TO_TEST,
    SAMPLE_SIZE
)

os.makedirs(EVAL_DIR, exist_ok=True)

def load_data():
    """Loads matching books from Library and Golden Dataset."""
    if not os.path.exists(GOLDEN_PATH):
        print(f"❌ Error: {GOLDEN_PATH} not found. Please create it first.")
        return []

    with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
        
    with open(LIBRARY_PATH, 'r', encoding='utf-8') as f:
        library_data = json.load(f)

    # Lookup map: Title -> Book Object
    library_map = {}
    for b in library_data:
        title = None
        if 'goodreads' in b:
            title = b['goodreads'].get('title') or b['goodreads'].get('Title')
        if not title:
            title = b.get('title') or b.get('Title')
        if title:
            try:
                library_map[title.strip()] = Book.from_dict(b)
            except Exception:
                pass
    
    all_matches = []
    print(f"📊 Loading Golden Dataset ({len(golden_data)} records)...")
    
    for title, truth in golden_data.items():
        book = library_map.get(title.strip())
        if book:
            if not book.google.description:
                print(f"   ⚠️  WARNING: Golden Book '{title}' has NO DESCRIPTION.")
            all_matches.append((book, truth))
        else:
            print(f"   ⚠️  Warning: Book '{title}' NOT found in Library. Skipping.")
            
    if len(all_matches) > SAMPLE_SIZE:
        print(f"🎲 Randomly selecting {SAMPLE_SIZE} books...")
        return random.sample(all_matches, SAMPLE_SIZE)
    
    print(f"✅ Loaded {len(all_matches)} golden books.")
    return all_matches

def check_match(ai_val, gold_val):
    """Boolean scoring logic. Robustly handles list vs string comparisons."""
    # If AI returned None, '', or [], it's an automatic fail
    if not ai_val: 
        return False 
        
    # If the Golden answer is a list, do an intersection check
    if isinstance(gold_val, list):
        # Force AI answer into a list if it returned a raw string
        if not isinstance(ai_val, list):
            ai_val = [ai_val] 
        # Pass if there is ANY overlap
        return len(set(ai_val) & set(gold_val)) > 0
        
    # Standard string comparison
    return str(ai_val).strip() == str(gold_val).strip()

def run_eval_arena():
    print(f"🥊 STARTING GOLDEN EVAL ARENA: {', '.join(MODELS_TO_TEST)}")
    
    test_data = load_data()
    if not test_data:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(timestamp)

    # The Master Data Structure (Model)
    eval_record = {
        "metadata": {
            "timestamp": timestamp,
            "models_tested": MODELS_TO_TEST,
            "books_tested": len(test_data),
            "fields_per_book": 5,
            "accuracy_scores": {m: 0.0 for m in MODELS_TO_TEST}
        },
        "results": []
    }

    # Pre-build the book entries in the results list
    for book, truth in test_data:
        eval_record["results"].append({
            "title": book.goodreads.title,
            "author": book.goodreads.author,
            "models": {}
        })
        
    model_points = {m: 0 for m in MODELS_TO_TEST}
    total_possible_points = len(test_data) * 5

    for model in MODELS_TO_TEST:
        print(f"\n🧠 LOADING MODEL: {model} (Running all books...)")
        
        for index, (book, truth) in enumerate(test_data):
            print(f"   📘 Evaluating ({index+1}/{len(test_data)}): {book.goodreads.title[:40]}...", end="\r")
            
            try:
                # RUN PIPELINE
                data = enrich_book(book, model=model)
                
                # SCORE FIELDS
                scores = {
                    "category": check_match(data.get('category'), truth['category']),
                    "genre": check_match(data.get('genre'), truth['genre']),
                    "sphere": check_match(data.get('sphere'), truth['sphere']),
                    "era": check_match(data.get('era'), truth['era']),
                    "location": check_match(data.get('location'), truth['location'])
                }
                
                points_earned = sum(scores.values())
                model_points[model] += points_earned

                # Inject data into the pre-built list based on index
                eval_record["results"][index]["models"][model] = {
                    "status": "success",
                    "fields": {
                        "category": {"ai": data.get('category'), "gold": truth['category'], "correct": scores["category"], "reasoning": data.get('cat_reasoning', '')},
                        "genre": {"ai": data.get('genre'), "gold": truth['genre'], "correct": scores["genre"], "reasoning": data.get('genre_reasoning', '')},
                        "sphere": {"ai": data.get('sphere'), "gold": truth['sphere'], "correct": scores["sphere"], "reasoning": data.get('sphere_reasoning', '')},
                        "era": {"ai": data.get('era'), "gold": truth['era'], "correct": scores["era"], "reasoning": data.get('era_reasoning', '')},
                        "location": {"ai": data.get('location'), "gold": truth['location'], "correct": scores["location"], "reasoning": data.get('loc_reasoning', '')}
                    },
                    "metrics": data.get('metrics', {})
                }

            except Exception as e:
                eval_record["results"][index]["models"][model] = {"status": "error", "error_msg": str(e)}
        
        print(f"   ✅ Done with all books for {model}.                ")

    # Calculate final accuracy percentages
    for m in MODELS_TO_TEST:
        eval_record["metadata"]["accuracy_scores"][m] = (model_points[m] / total_possible_points) * 100 if total_possible_points > 0 else 0

    # SAVE JSON
    json_path = os.path.join(EVAL_DIR, f"eval_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_record, f, indent=4)
    print(f"\n💾 Saved raw JSON data to: {json_path}")

    # TRIGGER HTML GENERATION
    generate_html_report(eval_record, timestamp)


def format_html_field(field_data):
    """Creates the Green/Red visual diffing for a single field."""
    
    # Safely handle AI values
    ai_raw = field_data.get('ai')
    if isinstance(ai_raw, list):
        ai_val = ", ".join(ai_raw) if ai_raw else "[Empty]"
    else:
        ai_val = str(ai_raw).strip() if ai_raw else "[Empty]"

    # Safely handle Golden values
    gold_raw = field_data.get('gold')
    gold_val = ", ".join(gold_raw) if isinstance(gold_raw, list) else str(gold_raw)
    
    reasoning_html = f"<br><i style='color:#555; font-size:0.85em;'>📝 {field_data['reasoning']}</i>" if ADD_REASONING_OUTPUT and field_data.get('reasoning') else ""
    
    if field_data['correct']:
        return f"<div style='margin-bottom:8px;'><span style='color:#2e7d32;'>✅ <b>{ai_val}</b></span>{reasoning_html}</div>"
    else:
        return (
            f"<div style='margin-bottom:8px;'><span style='color:#c62828;'>❌ <del>{ai_val}</del></span><br>"
            f"<span style='font-size:0.85em; background-color:#ffebee; padding:2px;'>🎯 {gold_val}</span>{reasoning_html}</div>"
        )

def generate_html_report(eval_record, timestamp):
    """The View: Builds HTML from the JSON data dict."""
    html_rows = []
    
    for book in eval_record["results"]:
        row = {"Title": f"<b>{book['title']}</b><br><span style='font-size:0.8em; color:#666'>{book['author']}</span>"}
        
        for model in eval_record["metadata"]["models_tested"]:
            model_data = book["models"].get(model, {})
            
            if model_data.get("status") == "error":
                row[model] = f"⚠️ CRASH<br>{model_data.get('error_msg')}"
                continue
            elif not model_data:
                row[model] = "N/A"
                continue

            fields = model_data["fields"]
            metrics = model_data.get("metrics", {})
            
            cell_content = [
                format_html_field(fields["category"]),
                format_html_field(fields["genre"]),
                format_html_field(fields["sphere"]),
                format_html_field(fields["era"]),
                format_html_field(fields["location"])
            ]
            
            if ADD_STEP_TIMING and metrics:
                cell_content.append(
                    f"<div style='margin-top:8px; font-size:0.75em; color:#888; border-top:1px dashed #ddd; padding-top:4px;'>"
                    f"⏱️ {metrics.get('total_time', 0)}s"
                    f"</div>"
                )
                
            row[model] = "".join(cell_content)
            
        html_rows.append(row)

    # Formatting Top Scores
    score_summary = ""
    for m, score in eval_record["metadata"]["accuracy_scores"].items():
        badge_color = "#4CAF50" if score > 85 else "#FF9800" if score > 70 else "#f44336"
        score_summary += f"<span style='display:inline-block; margin-right: 15px; padding: 5px 10px; border-radius: 15px; background-color: {badge_color}; color: white; font-weight: bold;'>{m}: {score:.1f}%</span>"

    df = pd.DataFrame(html_rows)
    
    style = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f4f9; }
        h2 { color: #333; margin-bottom: 10px; }
        .score-banner { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        th, td { border: 1px solid #eee; padding: 12px; vertical-align: top; text-align: left; }
        th { background-color: #2c3e50; color: white; font-weight: 600; }
        tr:nth-child(even) { background-color: #fafafa; }
    </style>
    """
    
    header = f"<h2>🎯 AI Model Benchmark (Golden Dataset)</h2><div class='score-banner'>{score_summary}</div>"
    html_output = style + header + df.to_html(escape=False, index=False)
    
    html_path = os.path.join(EVAL_DIR, f"eval_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_output)
        
    print(f"🚀 HTML report saved to: {html_path}")
    webbrowser.open('file://' + os.path.abspath(html_path))

if __name__ == "__main__":
    run_eval_arena()