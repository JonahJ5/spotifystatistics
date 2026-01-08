**Spotify Statistics Dashboard (Extended Streaming History)**

**Purpose:**
     This project allows users upload their Spotify Extended Streaming History export (my_spotify_data.zip) and generates descriptive analytics + visualizations in a simple dashboard.

**How to use:**
     
     1. Request your Extended Streaming History from Spotify: https://www.spotify.com/us/account/privacy/
          - It can take a few days to arrive.
     
     2. Open the app: https://spotifystatistics.streamlit.app/
     
     3. Upload the file my_spotify_data.zip (the ZIP you receive from Spotify).

**What data is in extended streaming data?**
(directly from the README Spotify provides in the ZIP)
| Technical field | Contains |
|---|---|
| `ts` | Timestamp indicating when the track stopped playing in UTC (year-month-day + 24h time). |
| `username` | Your Spotify username. |
| `platform` | Platform used when streaming (e.g., Android OS, Google Chromecast). |
| `ms_played` | Number of milliseconds the stream was played. |
| `conn_country` | Country code where the stream was played (e.g., SE = Sweden). |
| `ip_addr_decrypted` | IP address logged when streaming. |
| `user_agent_decrypted` | User agent used when streaming (e.g., a browser like Mozilla Firefox or Safari). |
| `master_metadata_track_name` | Name of the track. |
| `master_metadata_album_artist_name` | Name of the artist/band/podcast. |
| `master_metadata_album_album_name` | Name of the album for the track. |
| `spotify_track_uri` | Spotify track URI in the form `spotify:track:<base-62 string>` (can be searched in Spotify to locate the track). |
| `episode_name` | Name of the podcast episode. |
| `episode_show_name` | Name of the podcast show. |
| `spotify_episode_uri` | Spotify episode URI in the form `spotify:episode:<base-62 string>` (can be searched in Spotify to locate the episode). |
| `reason_start` | Value describing why the track started (e.g., `trackdone`). |
| `reason_end` | Value describing why the track ended (e.g., `endplay`). |
| `shuffle` | `True`/`False` depending on whether shuffle mode was used. |
| `skipped` | Indicates if the user skipped to the next song. |
| `offline` | Whether the track was played offline (`True`) or not (`False`). |
| `offline_timestamp` | Timestamp of when offline mode was used (if used). |
| `incognito_mode` | Whether the track was played during a private session (`True`) or not (`False`). |





**These are the only fields used to generate the dashboard:**

| Spotify field | Renamed in app | How it’s used |
|---|---|---|
| `ts` | `played_at` | Creates year/month/day/hour/day-of-week groupings and sessions |
| `ms_played` | `ms_played` | Converted into minutes for all listening-time metrics |
| `master_metadata_track_name` | `track` | Track rankings + repeat distribution |
| `master_metadata_album_artist_name` | `artist` | Artist rankings, trends, top-artist-per-day, artist “peak day” ranking |
| `master_metadata_album_album_name` | `album` | Album rankings |



**Not used for charts: username, ip_addr_decrypted, user_agent_decrypted** 


**Derived fields:**

     The app creates these fields from played_at (ts) and ms_played:
     
     - minutes = ms_played / 60000
     
     - day = calendar day (UTC)
     
     - month = YYYY-MM (UTC)
     
     - year = year (UTC) (used for the Year dropdown)
     
     - hour = hour of day (UTC)
     
     - dow = day of week (UTC)

**Sessions:**

     The “Sessions & Behavior” tab infers sessions from timestamps:
     
     A new session starts when the time gap between plays is greater than 30 minutes

**Sessions include:**

     - session_date
     
     - plays (# of play events)
     
     - minutes (total minutes in session)
     
     - duration_minutes
     
     - session_start / session_end (used for hover details)

**Data filtering:**

     To keep results consistent, the app filters the dataset:
     
     Only uses audio streaming history JSON files (typically named like Streaming_History_Audio_*.json)
     
     Requires:
     
     - ms_played > 0
     
     - non-empty track, artist, and album
     
     - Timezone: all groupings are based on UTC

**Dashboard features**

     - Rankings: top artists, tracks, albums, and repeat distribution
     
     - Time Patterns: heatmap by day-of-week/hour + top listening days (with that day’s top artist)
     
     - Trends: daily minutes, monthly minutes, cumulative minutes, top artists over time, artist diversity, new artists discovered
     
     - Sessions & Behavior: session distributions + plays vs minutes scatter (hover shows session date)

**What does “Show preview table” do?**

     If enabled, it displays the first 100 rows of your filtered dataset. This allows users to see what data is being used.

**Privacy note**

     This app does not require Spotify login and does not connect to your Spotify account. It analyzes the ZIP you upload during your session and generates visualizations from it. 

