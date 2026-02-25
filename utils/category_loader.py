import json
import os
from typing import Dict, List, Any
from config import LOCATIONS_PATH, SPHERES_PATH, GENRES_PATH, ERAS_PATH

# --- CONFIGURATION ---
CONFIG_DIR = "config"

def load_all_configs() -> Dict[str, Any]:
    """Loads all JSON files using the paths defined in config.py."""
    configs = {}
    
    # Mapping the internal data keys to the actual file paths
    # This removes the need for .replace(".json", "") logic
    files_map = {
        "eras": ERAS_PATH,
        "locations": LOCATIONS_PATH,
        "spheres": SPHERES_PATH,
        "top_level_genres": GENRES_PATH
    }
    
    for key, path in files_map.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                configs[key] = json.load(f)
        else:
            # Helpful for debugging the specific file that is missing
            print(f"Warning: Configuration file not found at {path}")
            configs[key] = {}
            
    return configs

# Load these once when the script starts
DATA = load_all_configs()

# --- HELPER FUNCTIONS ---

# Initialize the global DATA dictionary
DATA = load_all_configs()

# --- HELPER FUNCTIONS ---

def get_genres(main_category: str) -> List[str]:
    # Logic is sound, using the key from files_map
    key = "Fiction" if main_category.lower() == "fiction" else "Non-fiction"
    return DATA.get("top_level_genres", {}).get(key, [])

def get_spheres() -> List[str]:
    return list(DATA.get("spheres", {}).keys())

def get_spheres_with_descriptions() -> str:
    spheres_dict = DATA.get("spheres", {})
    formatted_lines = [f"- {name}: {desc.get('description', '') if isinstance(desc, dict) else desc}" 
                      for name, desc in spheres_dict.items()]
    return "\n".join(formatted_lines)
    
def get_eras(sphere: str) -> Dict[str, List[str]]:
    return DATA.get("eras", {}).get(sphere, {})

def get_era_definitions(sphere: str) -> str:
    sphere_eras = get_eras(sphere)
    formatted_lines = [f"- {broad_era}: Includes specific periods like [{', '.join(sub_eras)}]"
                      for broad_era, sub_eras in sphere_eras.items()]
    return "\n".join(formatted_lines)

def get_all_location_leaves() -> List[str]:
    """
    Returns a flattened list of all specific cities/regions (the 'leaves')
    using the already loaded DATA dictionary.
    """
    locations_data = DATA.get("locations", {})
    all_leaves = []
    # Level 1: Sphere -> Level 2: Region -> Level 3: City List
    for sphere_content in locations_data.values():
        for city_list in sphere_content.values():
            all_leaves.extend(city_list)
    
    return sorted(list(set(all_leaves)))

def validate_era(sphere: str, selection: str) -> bool:
    """Verifies that the choice is one of the Broad Eras (the Keys)."""
    sphere_eras = get_eras(sphere)
    # We only want the top-level keys
    valid_keys = list(sphere_eras.keys())
    return selection in valid_keys
