"""
Configuration and constants for ClipVid.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Video processing configuration
SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
AUDIO_SAMPLE_RATE = 16000  # 16kHz is optimal for speech recognition

# Chunking configuration
DEFAULT_CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "600"))
DEFAULT_CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
LONG_TRANSCRIPT_THRESHOLD_WORDS = 750

# Provider settings
DEFAULT_PROVIDER = "groq"

# Sanitize GROQ_API_KEY to handle surrounding whitespace or quotes
raw_groq_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = raw_groq_key.strip().strip('"').strip("'").strip()

# Default models
MODELS = {
    "groq": {
        "stt": "whisper-large-v3",
        "llm": "llama-3.3-70b-versatile"
    },
    "offline": {
        "stt": "built-in-mock",
        "llm": "built-in-heuristic"
    }
}
