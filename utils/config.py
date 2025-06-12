# app/utils/config.py
import os
import toml
from pathlib import Path

APP_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = APP_ROOT / "uploaded_files"
METADATA_DIR = APP_ROOT / "metadata"
CONFIG_FILE = APP_ROOT / "config.toml"

UPLOAD_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)


def get_gemini_api_key():
    # Load from config file first
    if CONFIG_FILE.exists():
        try:
            config = toml.load(CONFIG_FILE)
            return config.get("gemini", {}).get("api_key")
        except Exception as e:
            print(f"Error reading config.toml: {e}") # For debugging
    # Fallback to environment variable (optional, but good practice)
    return os.getenv("GOOGLE_API_KEY") # Use common GOOGLE_API_KEY env var as a secondary fallback

def save_gemini_api_key_to_config(api_key):
    config = {}
    if CONFIG_FILE.exists():
        try:
            config = toml.load(CONFIG_FILE)
        except Exception as e:
            print(f"Warning: Could not load existing config.toml to update, creating new. Error: {e}")

    if "gemini" not in config:
        config["gemini"] = {}

    config["gemini"]["api_key"] = api_key

    try:
        with open(CONFIG_FILE, "w") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"Error writing to config.toml: {e}") # For debugging
        return False