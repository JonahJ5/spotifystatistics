from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HELP_IMG = ASSETS_DIR / "spotify_directions.png"


# -----------------------------
# App constants
# -----------------------------
MAX_ZIP_MB = 300
MAX_FILES = 500
DEFAULT_TOPN = 15
SESSION_GAP_MINUTES = 30
MIN_STREAM_SECONDS = 30
MIN_STREAM_MS = MIN_STREAM_SECONDS * 1000


TIMEZONE_OPTIONS = {
    "Pacific Time": "America/Los_Angeles",
    "Mountain Time": "America/Denver",
    "Central Time": "America/Chicago",
    "Eastern Time": "America/New_York",
    "UTC": "UTC",
}
