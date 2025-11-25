"""
Configuration settings for the Hindi Educational Manhwa Content Generator.
Contains directory names and Gemini model rate-limit configurations.
"""

from pathlib import Path

# Output directories
OUTPUT_DIR = "manhwa_content"
METADATA_DIR = "manhwa_metadata"
CONTEXT_DIR = "chapter_context"

# Ensure directories exist when modules import these settings
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(METADATA_DIR).mkdir(exist_ok=True)
Path(CONTEXT_DIR).mkdir(exist_ok=True)

# Gemini Model Configuration with Rate Limits (Free Tier)
GEMINI_MODELS = {
    'gemini-2.0-flash-lite': {'rpm': 30, 'tpm': 1_000_000, 'rpd': 200},
    'gemini-2.0-flash-exp': {'rpm': 15, 'tpm': 1_000_000, 'rpd': 200},
    'gemini-2.5-flash': {'rpm': 10, 'tpm': 250_000, 'rpd': 250},
    'gemini-2.5-flash-lite': {'rpm': 15, 'tpm': 250_000, 'rpd': 1000},
    'gemini-2.5-flash-lite-preview-09-2025': {'rpm': 15, 'tpm': 250_000, 'rpd': 1000},
}

# Default model
DEFAULT_MODEL = 'gemini-2.5-flash-lite-preview-09-2025'
