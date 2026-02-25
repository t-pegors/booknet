import streamlit as st
import pandas as pd
import json
import os
import time
from typing import List, Any

from utils.goodreads_csv_extractor import main as run_goodreads_import, Book
from utils.googlebooks_enhancer import process_books as run_google_fetch
from utils.ai_enrichment import process_books as run_ai_enrichment
from utils.goodreads_scraper import process_books as run_goodreads_scrape
from utils.category_loader import get_genres, get_spheres, get_eras, get_all_location_leaves
from config import LIBRARY_PATH, OLLAMA_MODEL

# --- CONFIGURATION ---
PAGE_TITLE = "Goodreads Data Enrichment"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# THE GOLDEN CONFIG
# these are the fields required for a 100% completion score (other columns can be added
# in the flatten_books function)
GOLDEN_FIELDS = [
    "goodreads.title",
    "google.title",
    "goodreads.author",
    "google.authors",
    "goodreads.original_pub_year",
    "google.published_date", 
    "goodreads.description",
    "google.description",
    "google.thumbnail_url" 
]

COLUMN_CONFIGS = {
    # --- METADATA ---
    "cover_preview": st.column_config.ImageColumn("Cover", width="small"),
    "_status": st.column_config.TextColumn("State", width="small"), # Renamed from Status
    "_score": st.column_config.ProgressColumn(
        "Completeness", 
        min_value=0, max_value=100, format="%d%%", # Integer scale 0-100
    ),
    # --- GOODREADS (Source A) ---
    "goodreads_title": st.column_config.TextColumn("Title (GR)", width="medium"),
    "goodreads_author": st.column_config.TextColumn("Author", width="small"),
    "goodreads_original_pub_year": st.column_config.NumberColumn("Year", format="%d", width="small"),
    "goodreads_bookshelves": st.column_config.TextColumn("Shelves", width="medium"),
    "goodreads_description": st.column_config.TextColumn("Desc (GR)", width="large"),

    # --- GOOGLE (Source B) ---
    "google_title": st.column_config.TextColumn("Title (Google)", width="medium"),
    "google_subtitle": st.column_config.TextColumn("Subtitle", width="medium"),
    "google_description": st.column_config.TextColumn("Desc (Google)", width="large"),
    "google_categories": st.column_config.TextColumn("Categories", width="medium"),
    "google_thumbnail_url": st.column_config.TextColumn("Cover URL", width="large"),
}

# --- IMPORT DATA STRUCTURES ---
try:
    from utils.goodreads_csv_extractor import Book
except ImportError:
    import sys
    sys.path.append('utils')
    from goodreads_csv_extractor import Book

# --- HELPER FUNCTIONS ---

def load_data():
    if not os.path.exists(LIBRARY_PATH): return []
    with open(LIBRARY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Book.from_dict(item) for item in data]

def get_nested_value(obj: Any, path: str) -> Any:
    """Dynamically retrieves value like book.goodreads.title"""
    try:
        parts = path.split('.')
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        return None

def set_nested_value(obj: Any, path: str, value: Any):
    """Dynamically sets value like book.goodreads.title = 'Dune'"""
    parts = path.split('.')
    target = obj
    # Navigate to the parent of the final field
    for part in parts[:-1]:
        target = getattr(target, part)
    # Set the value on the final field
    setattr(target, parts[-1], value)

def flatten_books(books: List[Book]) -> pd.DataFrame:
    rows = []
    
    # Define the fields that are evaluated in pairs
    EITHER_OR_FIELDS = [
        "goodreads.description", "google.description",
        "goodreads.title", "google.title",
        "goodreads.author", "google.authors",
        "goodreads.original_pub_year", "google.published_date"
    ]
    
    for book in books:
        row = {"id": book.goodreads.id}
        filled_count = 0
        
        # Dynamic Field Extraction
        for field in GOLDEN_FIELDS:
            col_name = field.replace('.', '_')
            val = get_nested_value(book, field)
            
            # Lists -> CSV String
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val)
            
            row[col_name] = val
            
            # Standard Scoring: Check if "Truthy" but SKIP the paired fields!
            if field not in EITHER_OR_FIELDS:
                if val: filled_count += 1

        #### "EITHER/OR" LOGIC ###
        
        # Pair A: Descriptions
        if row.get("google_description") or row.get("goodreads_description"):
            filled_count += 1
            
        # Pair B: Authors
        if row.get("google_authors") or row.get("goodreads_author"):
            filled_count += 1
            
        # Pair C: Published Dates
        if row.get("google_published_date") or row.get("goodreads_original_pub_year"):
            filled_count += 1

        # Pair D: Titles
        if row.get("google_title") or row.get("goodreads_title"):
            filled_count += 1

        # NON-SCORED FIELDS
        row["google_subtitle"] = book.google.subtitle
        cats = book.google.categories
        row["google_categories"] = ", ".join(cats) if isinstance(cats, list) else str(cats) if cats else ""
        shelves = book.goodreads.bookshelves
        row["goodreads_bookshelves"] = ", ".join(shelves) if isinstance(shelves, list) else str(shelves) if shelves else ""
        
        # Add Computed Columns
        thumb = row.get("google_thumbnail_url", "")
        row["cover_preview"] = thumb if thumb and thumb.startswith("http") else None
        
        # Total golden fields minus the paired fields plus the points those pairs represent
        max_possible_points = 5 
        
        # Calculate percentage (0-100)
        score = int((filled_count / max_possible_points) * 100)
        
        # Cap at 100 just in case
        if score > 100: score = 100
            
        row["_score"] = score
        
        # Clean Status Text (Using 0-100 scale)
        if score == 100: row["_status"] = "COMPLETE"
        elif score >= 50: row["_status"] = "PARTIAL"
        else: row["_status"] = "EMPTY"

        rows.append(row)
        
    return pd.DataFrame(rows)

def save_data(edited_df: pd.DataFrame, original_books: List[Book]):
    book_map = {b.goodreads.id: b for b in original_books}
    
    for _, row in edited_df.iterrows():
        book = book_map.get(row["id"])
        if not book: continue
            
        # Dynamically map DataFrame columns back to Nested Object
        for field in GOLDEN_FIELDS:
            col_name = field.replace('.', '_')
            new_val = row.get(col_name)
            
            # --- TYPE CONVERSION LOGIC ---
            
            # 1. Handle Lists (CSV String -> List)
            if any(x in field for x in ['categories', 'tags', 'bookshelves']):
                if new_val and isinstance(new_val, str):
                    new_val = [x.strip() for x in new_val.split(',') if x.strip()]
                elif not new_val:
                    new_val = []

            # 2. Handle Numbers (String -> Int/None)
            elif 'year' in field or 'count' in field or 'rating' in field:
                try:
                    new_val = int(float(new_val)) if new_val and new_val != "" else None
                except:
                    new_val = None
                    
            # 3. Default: Strings
            else:
                new_val = str(new_val) if new_val is not None else ""

            # Save back to object
            set_nested_value(book, field, new_val)

    # Write to Disk
    with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
        json.dump([b.to_dict() for b in original_books], f, indent=4)
    
    st.toast("✅ Saved!", icon="💾")

###############################################################################################################
#### MAIN UI ####
###############################################################################################################

if 'library' not in st.session_state:
    st.session_state.library = load_data()

banner_path = "assets/data_dashboard/banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, width=800)
else:
    st.warning("Banner asset missing, skipping image render.")

st.title(PAGE_TITLE)
st.caption("This dashboard is used to load and enrich Goodreads export data.")
# Clean Tab Labels
tab1, tab2, tab3, tab4 = st.tabs(["Ingestion", "Curation", "Enrichment", "Validation"])

with tab1:
    
    st.subheader("1. Goodreads Import")
    st.caption("Pull all new data from the most recent Goodreads CSV file.")
    
    if st.button("Sync with latest Goodreads CSV"):
        with st.spinner("Extracting and Merging..."):
            stats = run_goodreads_import() # Runs your CLI wrapper
            st.session_state.library = load_data() # Reload fresh data into memory
        
        # Display the stats if it succeeded
        if stats:
            st.success(f"✅ Goodreads sync complete! Found **{stats['total_read']}** read books. Added **{stats['new_added']}** new books to the library.")
        else:
            st.error("❌ Sync failed. Could not find a matching CSV file.")

    st.divider()

    st.subheader("2. Google Books Metadata")
    st.caption("Grabs all Google Books data associated with Goodreads books.")
    if st.button("Fetch Cover Image URLs, Descriptions, etc."):
        # Setup empty UI elements for the callback to fill
        progress_bar = st.progress(0)
        status_text = st.empty()
        timer_text = st.empty()
        
        # Start the clock
        start_time = time.time()

        def update_progress(index, total, book):
            progress = (index + 1) / total
            progress_bar.progress(progress)
            
            # --- TIMING LOGIC ---
            elapsed_time = time.time() - start_time
            books_done = index + 1
            avg_time = elapsed_time / books_done
            eta_seconds = avg_time * (total - books_done)
            
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            # Update UI
            status_text.text(f"Fetching: {book.goodreads.title}")
            timer_text.caption(f"⏱️ **Elapsed:** {elapsed_str} | **ETA:** {eta_str} ({avg_time:.1f}s/book)")
        
        # Run the core logic in memory!
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        run_google_fetch(st.session_state.library, api_key=api_key, progress_callback=update_progress)
        
        # Save the in-memory updates to disk
        with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
            json.dump([b.to_dict() for b in st.session_state.library], f, indent=4)
        
        total_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        status_text.success(f"✅ Google metadata fully updated! Total time: {total_time}")
        timer_text.empty()
        
    st.divider()

    st.subheader("3. Scrape Additional Goodreads Data")
    st.caption("Scrapes the Goodreads website for more detailed descriptions and metadata.")
    if st.button("Scrape Missing Goodreads Info"):
        # Setup UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        timer_text = st.empty() # Container for the clock
        
        # Start the clock
        start_time = time.time()
        
        def update_progress(index, total, book):
            progress = (index + 1) / total
            progress_bar.progress(progress)
            
            # --- TIMING LOGIC ---
            elapsed_time = time.time() - start_time
            books_done = index + 1
            avg_time = elapsed_time / books_done
            eta_seconds = avg_time * (total - books_done)
            
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            # Update UI
            status_text.text(f"Scraping: {book.goodreads.title}")
            timer_text.caption(f"⏱️ **Elapsed:** {elapsed_str} | **ETA:** {eta_str} ({avg_time:.1f}s/book)")
        
        # Run the core scraping logic
        run_goodreads_scrape(st.session_state.library, progress_callback=update_progress)
        
        # Save updates to disk
        with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
            json.dump([b.to_dict() for b in st.session_state.library], f, indent=4)
            
        # Finalization
        total_time_raw = time.time() - start_time
        total_time = time.strftime('%H:%M:%S', time.gmtime(total_time_raw))
        status_text.success(f"✅ Goodreads scraping complete! Total time: {total_time}")
        timer_text.empty()

with tab2:
    
    if not st.session_state.library:
        st.info("Nothing to curate yet.")
    else:
        df = flatten_books(st.session_state.library)

        # Top Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Books", len(st.session_state.library))
        c2.metric("🟢 Complete", len(df[df["_score"] == 100]))
        c3.metric("🟡 Partial", len(df[(df["_score"] >= 50) & (df["_score"] < 100)]))
        c4.metric("🔴 Incomplete", len(df[df["_score"] < 50]))

        # Filter and Legend
        st.divider()

        # Create two columns: Left for the Filter, Right for the Legend
        col_filter, col_legend = st.columns([1, 2])

        with col_filter:
            show_incomplete = st.checkbox("Show only incomplete books")

        with col_legend:
            st.caption("GR = Goodreads Source | Google = Google Books Source")

        # Apply Filter Logic
        if show_incomplete:
            # Filter the dataframe but keep the original 'books' list intact for saving
            df = df[df["_score"] < 100]

        # Define Display Order
        # This ensures the visual grouping of Blue/Search columns
        display_order = [
            "_status", "_score", "cover_preview", 
            "goodreads_title", "google_title", "google_subtitle",
            "goodreads_author", "google_authors",
            "goodreads_original_pub_year", "google_published_date",
            "goodreads_bookshelves", "google_categories", 
            "goodreads_description", "google_description", 
            "google_thumbnail_url"
        ]

        # Render Editor
        edited_df = st.data_editor(
            df,
            column_config=COLUMN_CONFIGS,
            column_order=display_order,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            height=600 
        )

        if st.button("💾 Save Changes", type="primary"):
            # Save against the session state library!
            save_data(edited_df, st.session_state.library)
            st.rerun()

with tab3:
    st.header("AI Enrichment Pipeline")
    if not st.session_state.library:
        st.info("Ingest some books before running the AI!")
    else:
        st.write(f"Active Model: **{OLLAMA_MODEL}**")
        
        force_ai_update = st.checkbox(
            "Overwrite existing AI data", 
            value=False, 
            help="Force LLM to re-evaluate all books, ignoring any previous AI results."
        )

        if st.button("Run AI Pipeline"):
            progress_bar = st.progress(0)
            
            # Create two separate empty UI containers: one for the book data, one for the timer
            status_text = st.empty()
            timer_text = st.empty()
            
            # Record the exact moment the process starts
            start_time = time.time()
            
            def ai_progress(index, total, book):
                progress = (index + 1) / total
                progress_bar.progress(progress)
                
                # --- TIMING LOGIC ---
                elapsed_time = time.time() - start_time
                books_done = index + 1
                
                # Calculate the average time per book, then multiply by remaining books
                avg_time_per_book = elapsed_time / books_done
                eta_seconds = avg_time_per_book * (total - books_done)
                
                # Format times into HH:MM:SS for easy reading
                elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                # --- DISPLAY LOGIC ---
                # Safely format the location whether the LLM returned a list or a string
                loc = book.llm.location
                loc_str = ", ".join(loc) if isinstance(loc, list) else str(loc)
                
                # Build a rich markdown string of ALL the LLM data
                llm_details = (
                    f"**Category:** {book.llm.category} | "
                    f"**Genre:** {book.llm.genre} | "
                    f"**Sphere:** {book.llm.sphere} | "
                    f"**Era:** {book.llm.era} | "
                    f"**Location:** {loc_str}"
                )
                
                # Update the UI
                status_text.markdown(f"**Enriched:** {book.goodreads.title}  \n✨ {llm_details}")
                timer_text.caption(f"⏱️ **Elapsed:** {elapsed_str} | **Estimated Remaining:** {eta_str}  *(~{avg_time_per_book:.1f}s per book)*")
                
                # Save to disk every 10 books so we don't lose data if it crashes!
                if index > 0 and index % 10 == 0:
                    with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
                        json.dump([b.to_dict() for b in st.session_state.library], f, indent=4)

            # Run the Pipeline
            run_ai_enrichment(st.session_state.library, model=OLLAMA_MODEL, progress_callback=ai_progress, force_update=force_ai_update)
            
            # Save the final data
            with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
                json.dump([b.to_dict() for b in st.session_state.library], f, indent=4)
                
            # --- FINALIZATION ---
            # Calculate the total time taken from start to finish
            total_elapsed = time.time() - start_time
            total_str = time.strftime('%H:%M:%S', time.gmtime(total_elapsed))
            
            # Display final success message and clear the running timer
            status_text.success(f"✅ AI Enrichment completely finished! Total processing time: **{total_str}**")
            timer_text.empty()

with tab4:
    st.header("AI Validation & Ground Truth")
    
    if not st.session_state.library:
        st.info("Ingest some books before validating!")
    else:
        # 1. Build a flat dataframe specifically for the AI view
        ai_rows = []
        for b in st.session_state.library:
            ai_rows.append({
                "id": b.goodreads.id,
                "Title": b.goodreads.title,
                "Category": b.llm.category,
                "Genre": b.llm.genre,
                "Sphere": b.llm.sphere,
                "Era": b.llm.era,
                "Validated": b.llm.validated
            })
        
        df_ai = pd.DataFrame(ai_rows)
        
        st.markdown("### 1. Select a Book to Review")
        # 2. Render the interactive table
        event = st.dataframe(
            df_ai,
            column_config={
                "id": None, # Hide the ID column
                "Validated": st.column_config.CheckboxColumn("Validated")
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            height=250
        )
        
        # 3. THE FOCUS EDITOR
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_id = df_ai.iloc[selected_idx]["id"]
            
            # Find the actual book object in memory
            book = next((b for b in st.session_state.library if b.goodreads.id == selected_id), None)
            
            if book:
                st.divider()
                st.markdown(f"### 2. Validating: **{book.goodreads.title}**")
                
                # --- SNAPSHOT & ACCURACY LOGIC ---
                # If the AI ran but we haven't taken a snapshot yet, take it now!
                if not book.llm.ai_snapshot and book.llm.category:
                    book.llm.ai_snapshot = {
                        "category": book.llm.category,
                        "genre": book.llm.genre,
                        "sphere": book.llm.sphere,
                        "era": book.llm.era,
                        "location": book.llm.location
                    }
                
                # Calculate live accuracy score based on your edits
                if book.llm.ai_snapshot:
                    matches = 0
                    if book.llm.category == book.llm.ai_snapshot.get("category"): matches += 1
                    if book.llm.genre == book.llm.ai_snapshot.get("genre"): matches += 1
                    if book.llm.sphere == book.llm.ai_snapshot.get("sphere"): matches += 1
                    if book.llm.era == book.llm.ai_snapshot.get("era"): matches += 1
                    if sorted(book.llm.location) == sorted(book.llm.ai_snapshot.get("location", [])): matches += 1
                    
                    st.caption(f"🤖 **Mistral-Nemo Accuracy:** {matches}/5 fields match the original AI inference.")
                
                # --- THE EDITING FORM ---
                # We use a [1, 4] ratio to give the cover a fixed sidebar feel 
                # and the fields plenty of room to breathe.
                col_cover, col_right = st.columns([1, 4])
                
                with col_cover:
                    # The Framed Cover
                    if book.google.thumbnail_url:
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #E0E0E0; padding: 4px; border-radius: 10px; 
                                        background-color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                                <img src="{book.google.thumbnail_url}" style="width: 100%; border-radius: 5px; display: block;">
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """<div style="height: 200px; border: 2px dashed #E0E0E0; border-radius: 10px; 
                                        display: flex; align-items: center; justify-content: center; 
                                        color: #999; background-color: #F8F9FB;">No Cover Found</div>""", 
                            unsafe_allow_html=True
                        )

                with col_right:
                    
                    # Fetch all specific cities/sites from your enriched locations.json
                    loc_opts = get_all_location_leaves()
                    
                    # Clean the book's current locations: only keep ones that exist in loc_opts
                    # to prevent Streamlit from throwing an error on 'default' values
                    valid_locs = [l for l in book.llm.location if l in loc_opts]

                    # --- SOURCE DESCRIPTIONS (New Section) ---
                    gr_desc = getattr(book.goodreads, 'description', "")
                    go_desc = getattr(book.google, 'description', "")

                    if gr_desc or go_desc:
                        with st.expander("View Source Descriptions", expanded=True):
                            if gr_desc:
                                st.markdown(f"*[Goodreads]*: *{gr_desc}*")
                            if go_desc:
                                # Add a small divider if both exist
                                if gr_desc: st.write("---")
                                st.markdown(f"*[Google]*: *{go_desc}*")
                    
                    st.write("") # Spacer before fields

                    # Nested columns for the primary dropdowns
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        # Category
                        cat_opts = ["Fiction", "Non-fiction"]
                        cat_idx = cat_opts.index(book.llm.category) if book.llm.category in cat_opts else 0
                        new_cat = st.selectbox("Category", cat_opts, index=cat_idx, key=f"cat_{book.goodreads.id}")
                        
                        # Genre
                        genre_opts = get_genres(new_cat) or ["Unknown"]
                        genre_idx = genre_opts.index(book.llm.genre) if book.llm.genre in genre_opts else 0
                        new_genre = st.selectbox("Genre", genre_opts, index=genre_idx, key=f"gen_{book.goodreads.id}")
                    
                    with c2:
                        # Sphere
                        sphere_opts = get_spheres() or ["Unknown"]
                        sphere_idx = sphere_opts.index(book.llm.sphere) if book.llm.sphere in sphere_opts else 0
                        new_sphere = st.selectbox("Sphere", sphere_opts, index=sphere_idx, key=f"sph_{book.goodreads.id}")
                        
                        # Era
                        raw_eras = get_eras(new_sphere)
                        if raw_eras and isinstance(raw_eras, dict):
                            era_opts = sorted([e for sublist in raw_eras.values() for e in sublist])
                        else:
                            era_opts = ["Unknown"]

                        if book.llm.era and book.llm.era not in era_opts:
                            era_opts.insert(0, book.llm.era)

                        era_idx = era_opts.index(book.llm.era) if book.llm.era in era_opts else 0
                        new_era = st.selectbox("Era", era_opts, index=era_idx, key=f"era_{book.goodreads.id}")
                    
                    # Location (Now inside the right column area)
                    new_loc = st.multiselect("Location (Multiple allowed)", loc_opts, default=valid_locs, key=f"loc_{book.goodreads.id}")
                    
                    # Validation & Save Action
                    col_val, col_btn = st.columns([2, 1])
                    with col_val:
                        new_val = st.checkbox("✅ Mark as Validated (Human Reviewed)", value=book.llm.validated, key=f"val_{book.goodreads.id}")
                    
                    with col_btn:
                        if st.button("💾 Save Validation", type="primary", use_container_width=True):
                            # Update the in-memory object
                            book.llm.category = new_cat
                            book.llm.genre = new_genre
                            book.llm.sphere = new_sphere
                            book.llm.era = new_era
                            book.llm.location = new_loc
                            book.llm.validated = new_val
                            
                            # Save the entire library to disk
                            with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
                                json.dump([b.to_dict() for b in st.session_state.library], f, indent=4)
                                
                            st.success("Validation Saved!")
                            st.rerun()

                
        else:
            st.info("👆 Click on a row in the table above to open the validation editor.")