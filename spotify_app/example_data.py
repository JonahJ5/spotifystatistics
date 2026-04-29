import random

import pandas as pd

from spotify_app.config import MIN_STREAM_MS
from spotify_app.data_transform import add_time_fields


def make_example_data(n_rows: int = 2500) -> pd.DataFrame:
    """Create fake Spotify-style listening data so users can preview the dashboard."""
    rng = random.Random(42)

    artists = [
        "Arctic Monkeys", "SZA", "Kendrick Lamar", "Tame Impala", "Fleetwood Mac",
        "Drake", "Frank Ocean", "Taylor Swift", "The Weeknd", "Mac Miller",
        "Paramore", "Tyler, The Creator",
    ]

    tracks_by_artist = {
        "Arctic Monkeys": ["Do I Wanna Know?", "505", "R U Mine?", "Fluorescent Adolescent"],
        "SZA": ["Kill Bill", "Good Days", "Snooze", "Broken Clocks"],
        "Kendrick Lamar": ["HUMBLE.", "Money Trees", "DNA.", "Alright"],
        "Tame Impala": ["The Less I Know The Better", "Let It Happen", "Eventually", "Borderline"],
        "Fleetwood Mac": ["Dreams", "The Chain", "Go Your Own Way", "Landslide"],
        "Drake": ["Passionfruit", "God's Plan", "One Dance", "Headlines"],
        "Frank Ocean": ["Pink + White", "Nights", "Thinkin Bout You", "Lost"],
        "Taylor Swift": ["Cruel Summer", "Style", "Anti-Hero", "Blank Space"],
        "The Weeknd": ["Blinding Lights", "Save Your Tears", "Starboy", "Out of Time"],
        "Mac Miller": ["Good News", "Self Care", "Dang!", "Weekend"],
        "Paramore": ["Still Into You", "Hard Times", "Misery Business", "Ain't It Fun"],
        "Tyler, The Creator": ["See You Again", "EARFQUAKE", "WUSYANAME", "Sweet"],
    }

    artist_weights = [11, 10, 10, 9, 8, 8, 8, 8, 8, 7, 6, 5]

    hour_weights = [
        2, 1, 1, 1, 1, 2, 4, 6, 7, 6, 5, 5,
        6, 6, 7, 8, 9, 11, 13, 14, 12, 9, 6, 4,
    ]

    today = pd.Timestamp.now(tz="UTC").floor("D")
    start = today - pd.Timedelta(days=365)

    rows = []

    for _ in range(n_rows):
        artist = rng.choices(artists, weights=artist_weights, k=1)[0]
        track = rng.choice(tracks_by_artist[artist])
        album = f"{artist} Essentials"

        day_offset = rng.randrange(365)
        hour = rng.choices(list(range(24)), weights=hour_weights, k=1)[0]
        minute = rng.randrange(60)

        played_at = start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)
        ms_played = rng.randint(MIN_STREAM_MS, 240_000)

        rows.append({
            "played_at": played_at,
            "ms_played": ms_played,
            "track": track,
            "artist": artist,
            "album": album,
            "_source_file": "example_data",
        })

    example = pd.DataFrame(rows).sort_values("played_at").reset_index(drop=True)
    example["minutes"] = example["ms_played"] / 60000.0
    example = add_time_fields(example)

    return example
