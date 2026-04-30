# Spotify Statistics Dashboard

A Streamlit dashboard for exploring Spotify Extended Streaming History exports.

The app lets users upload the ZIP file from Spotify, then builds interactive listening statistics for artists, tracks, albums, time patterns, trends, repeat behavior, and listening sessions. If no ZIP is uploaded, the app shows generated example data so the dashboard can be previewed.

## Live App

https://spotifystatistics.streamlit.app/

## How To Use

1. Request your Spotify data from Spotify's privacy page:
   https://www.spotify.com/us/account/privacy/
2. Select **Extended streaming history** when requesting the export.
3. Wait for Spotify to prepare the ZIP file.
4. Open the app and upload the ZIP file Spotify provides.
5. Choose your timezone and filters in the sidebar.

The app does not require Spotify login and does not connect to your Spotify account.

## Dashboard Features

- Rankings for top artists, tracks, and albums by listening time and play count
- Track repeat distribution
- Artist peak days
- Hours by day of week and hour of day
- Day/hour listening heatmap
- Most-listened days with detail views
- Consecutively repeated songs
- Listening trends over time, defaulting to weekly view
- Cumulative listening hours
- Top artist trends over time
- Artist diversity and new artist discovery
- Session distributions and longest sessions
- Shareable dark-mode PDF snapshot
- Technical CSV and JSON exports

## Filters

The sidebar includes:

- **Top N**: controls how many ranked items are shown
- **Timezone**: controls local day, hour, week, month, and year groupings
- **Year**: quick filter for a single calendar year or all years
- **Advanced date range**: optional custom start/end date filter that overrides the Year filter
- **Show preview table**: shows the first 100 filtered rows

## Data Used

The app is designed for Spotify's **Extended Streaming History** export. It looks for audio streaming history JSON files inside the uploaded ZIP.

The dashboard uses these Spotify fields:

| Spotify field | App field | Used for |
|---|---|---|
| `ts` | `played_at` | Dates, hours, trends, and sessions |
| `ms_played` | `ms_played` | Listening-time metrics |
| `master_metadata_track_name` | `track` | Track rankings and repeat behavior |
| `master_metadata_album_artist_name` | `artist` | Artist rankings, trends, and diversity |
| `master_metadata_album_album_name` | `album` | Album rankings |

The app does not use private account fields such as username, IP address, or user agent for charts.

## Derived Fields

After loading the data, the app creates:

- `minutes`
- `played_at_local`
- `date`
- `day`
- `week`
- `month`
- `year`
- `hour`
- `hour_label`
- `day_of_week`

Timezone-sensitive fields are based on the selected timezone.

## Data Filtering

To keep charts focused on meaningful music plays, the loader requires:

- Valid track, artist, and album metadata
- Positive play duration
- At least the configured minimum stream length

## Sessions

The Sessions & Behavior tab infers listening sessions from play timestamps. A new session starts when the gap between plays is greater than the configured session gap.

Session charts use hours to avoid confusion between total listening time and elapsed session span.

## Exports

The app includes:

- A shareable dark-mode PDF snapshot of the current filtered dashboard
- `wrapped_summary.json`
- CSV exports for top artists, tracks, albums, daily minutes, and monthly minutes
- A ZIP download containing all technical exports

The PDF snapshot reflects the active timezone, year or custom date range, Top N setting, and dashboard filters.

## Privacy

Uploaded data is processed only in the active Streamlit session. The app does not store your Spotify ZIP, does not require OAuth, and does not send your data to Spotify or any external analytics service.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
