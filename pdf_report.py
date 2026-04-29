import pandas as pd
import plotly.express as px

from spotify_app.config import SESSION_GAP_MINUTES
from spotify_app.data_transform import compute_sessions, format_hour_label, safe_group_sum


def build_shareable_pdf(df: pd.DataFrame, topn: int) -> bytes:
    """Create a multi-page shareable PDF summary with one page per dashboard tab."""
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )

    def clean_text(value, max_len: int = 48) -> str:
        text = "" if pd.isna(value) else str(value)
        text = text.replace("\n", " ").strip()
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return text

    def add_table(story, rows, col_widths=None):
        if not rows:
            return

        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1DB954")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.16 * inch))

    def fig_to_rl_image(fig, width_inches=4.8, height_inches=2.7):
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=45),
            font=dict(size=10),
        )
        img_bytes = fig.to_image(
            format="png",
            width=int(width_inches * 300),
            height=int(height_inches * 300),
            scale=1,
        )
        return Image(
            BytesIO(img_bytes),
            width=width_inches * inch,
            height=height_inches * inch,
        )

    def add_chart_pair(story, fig_left, fig_right, width_inches=4.8, height_inches=2.6):
        left_img = fig_to_rl_image(fig_left, width_inches=width_inches, height_inches=height_inches)
        right_img = fig_to_rl_image(fig_right, width_inches=width_inches, height_inches=height_inches)

        chart_table = Table(
            [[left_img, right_img]],
            colWidths=[width_inches * inch, width_inches * inch],
        )
        chart_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(chart_table)
        story.append(Spacer(1, 0.14 * inch))

    def add_single_chart(story, fig, width_inches=10.0, height_inches=2.3):
        img = fig_to_rl_image(fig, width_inches=width_inches, height_inches=height_inches)
        story.append(img)
        story.append(Spacer(1, 0.14 * inch))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    date_source = "played_at_local" if "played_at_local" in df.columns else "played_at"
    start_date = pd.to_datetime(df[date_source]).min().strftime("%b %d, %Y")
    end_date = pd.to_datetime(df[date_source]).max().strftime("%b %d, %Y")

    total_minutes = df["minutes"].sum()
    total_hours = total_minutes / 60
    track_plays = len(df)
    unique_artists = df["artist"].nunique()
    unique_tracks = df["track"].nunique()
    unique_albums = df["album"].nunique()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour_order = [format_hour_label(h) for h in range(24)]

    # Page 1: Rankings
    story.append(Paragraph("Spotify Statistics Summary", styles["Title"]))
    story.append(Paragraph("Created with Spotify Statistics by Jonah Jutzi", styles["Normal"]))
    story.append(Paragraph(f"Listening history range: {start_date} to {end_date}", styles["Normal"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Rankings", styles["Heading1"]))

    kpi_rows = [
        ["Metric", "Value"],
        ["Total Hours", f"{total_hours:,.1f}"],
        ["Total Minutes", f"{total_minutes:,.0f}"],
        ["Track Plays", f"{track_plays:,}"],
        ["Unique Artists", f"{unique_artists:,}"],
        ["Unique Tracks", f"{unique_tracks:,}"],
        ["Unique Albums", f"{unique_albums:,}"],
    ]
    add_table(story, kpi_rows, col_widths=[2.4 * inch, 1.6 * inch])

    top_artists = safe_group_sum(df, ["artist"], "minutes", min(topn, 10)).copy()
    top_artists["hours"] = top_artists["minutes"] / 60.0
    fig_artists = px.bar(
        top_artists.sort_values("hours", ascending=True),
        x="hours",
        y="artist",
        orientation="h",
        title="Top Artists by Hours",
        labels={"hours": "Hours", "artist": "Artist"},
    )
    fig_artists.update_layout(showlegend=False)

    top_tracks = safe_group_sum(df, ["track", "artist", "album"], "minutes", min(topn, 10)).copy()
    top_tracks["hours"] = top_tracks["minutes"] / 60.0
    top_tracks["track_label"] = top_tracks["track"].apply(lambda x: clean_text(x, 28))
    fig_tracks = px.bar(
        top_tracks.sort_values("hours", ascending=True),
        x="hours",
        y="track_label",
        orientation="h",
        title="Top Tracks by Hours",
        labels={"hours": "Hours", "track_label": "Track"},
    )
    fig_tracks.update_layout(showlegend=False)

    add_chart_pair(story, fig_artists, fig_tracks, width_inches=4.9, height_inches=2.7)

    top_albums = safe_group_sum(df, ["album", "artist"], "minutes", min(topn, 8)).copy()
    top_albums["hours"] = top_albums["minutes"] / 60.0
    top_albums["album_label"] = top_albums["album"].apply(lambda x: clean_text(x, 35))
    fig_albums = px.bar(
        top_albums.sort_values("hours", ascending=True),
        x="hours",
        y="album_label",
        orientation="h",
        title="Top Albums by Hours",
        labels={"hours": "Hours", "album_label": "Album"},
    )
    fig_albums.update_layout(showlegend=False)

    add_single_chart(story, fig_albums, width_inches=10.0, height_inches=2.0)

    # Page 2: Time Patterns
    story.append(PageBreak())
    story.append(Paragraph("Time Patterns", styles["Heading1"]))
    story.append(Spacer(1, 0.08 * inch))

    by_dow = df.groupby("day_of_week", as_index=False)["minutes"].sum()
    by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=order, ordered=True)
    by_dow = by_dow.sort_values("day_of_week")
    by_dow["hours"] = by_dow["minutes"] / 60.0

    fig_dow = px.bar(
        by_dow,
        x="day_of_week",
        y="hours",
        title="Hours by Day of the Week",
        labels={"day_of_week": "Day of the Week", "hours": "Hours"},
    )
    fig_dow.update_layout(showlegend=False)

    by_hour = df.groupby(["hour", "hour_label"], as_index=False)["minutes"].sum().sort_values("hour")
    by_hour["hour_label"] = pd.Categorical(by_hour["hour_label"], categories=hour_order, ordered=True)
    by_hour = by_hour.sort_values("hour_label")
    by_hour["hours"] = by_hour["minutes"] / 60.0

    fig_hour = px.bar(
        by_hour,
        x="hour_label",
        y="hours",
        title="Hours by Hour of Day",
        labels={"hour_label": "Hour of Day", "hours": "Hours"},
    )
    fig_hour.update_layout(showlegend=False)

    add_chart_pair(story, fig_dow, fig_hour, width_inches=4.9, height_inches=2.6)

    repeat_sequence = df.sort_values("played_at").copy()
    repeat_sequence["track_key"] = (
        repeat_sequence["track"].astype(str) + " — " + repeat_sequence["artist"].astype(str)
    )
    repeat_sequence["new_streak"] = repeat_sequence["track_key"] != repeat_sequence["track_key"].shift()
    repeat_sequence["streak_id"] = repeat_sequence["new_streak"].cumsum()

    streaks = (
        repeat_sequence.groupby("streak_id", as_index=False)
        .agg(
            track=("track", "first"),
            artist=("artist", "first"),
            repeat_count=("track", "size"),
        )
    )
    streaks = streaks[streaks["repeat_count"] > 1].copy()
    streaks = streaks.sort_values("repeat_count", ascending=False).head(10)

    if not streaks.empty:
        streaks["label"] = (
            streaks["track"].apply(lambda x: clean_text(x, 28))
            + " — "
            + streaks["artist"].apply(lambda x: clean_text(x, 20))
        )

        fig_repeat = px.bar(
            streaks.sort_values("repeat_count", ascending=True),
            x="repeat_count",
            y="label",
            orientation="h",
            title="Most Consecutively Repeated Songs",
            labels={"repeat_count": "Consecutive Plays", "label": "Track"},
        )
        fig_repeat.update_layout(showlegend=False)
        add_single_chart(story, fig_repeat, width_inches=10.0, height_inches=2.2)

    # Page 3: Trends
    story.append(PageBreak())
    story.append(Paragraph("Trends", styles["Heading1"]))
    story.append(Spacer(1, 0.08 * inch))

    month_count = df["month"].nunique()
    trend_col = "month" if month_count <= 60 else "year"
    trend_label = "Month" if trend_col == "month" else "Year"

    listening_trend = (
        df.groupby(trend_col, as_index=False)["minutes"].sum()
        .sort_values(trend_col)
    )
    listening_trend["hours"] = listening_trend["minutes"] / 60.0

    fig_listening = px.line(
        listening_trend,
        x=trend_col,
        y="hours",
        title=f"Hours Over Time by {trend_label}",
        labels={trend_col: trend_label, "hours": "Hours"},
    )

    cumulative = listening_trend.copy()
    cumulative["cumulative_hours"] = cumulative["hours"].cumsum()

    fig_cumulative = px.line(
        cumulative,
        x=trend_col,
        y="cumulative_hours",
        title=f"Cumulative Hours Over Time by {trend_label}",
        labels={trend_col: trend_label, "cumulative_hours": "Cumulative Hours"},
    )

    add_chart_pair(story, fig_listening, fig_cumulative, width_inches=4.9, height_inches=2.8)

    diversity = (
        df.groupby(trend_col, as_index=False)
        .agg(unique_artists=("artist", "nunique"))
        .sort_values(trend_col)
    )

    fig_diversity = px.line(
        diversity,
        x=trend_col,
        y="unique_artists",
        title=f"Artist Diversity Over Time by {trend_label}",
        labels={trend_col: trend_label, "unique_artists": "Unique Artists"},
    )

    first_artist = (
        df.sort_values("played_at")
        .groupby("artist", as_index=False)
        .first()[["artist", "played_at"]]
    )

    if trend_col == "month":
        first_artist["period"] = first_artist["played_at"].dt.strftime("%Y-%m")
        discovery_label = "Month"
    else:
        first_artist["period"] = first_artist["played_at"].dt.year
        discovery_label = "Year"

    discovery = (
        first_artist.groupby("period", as_index=False)
        .size()
        .rename(columns={"size": "new_artists"})
        .sort_values("period")
    )

    fig_discovery = px.bar(
        discovery,
        x="period",
        y="new_artists",
        title=f"New Artists Discovered by {discovery_label}",
        labels={"period": discovery_label, "new_artists": "New Artists"},
    )
    fig_discovery.update_layout(showlegend=False)

    add_chart_pair(story, fig_diversity, fig_discovery, width_inches=4.9, height_inches=2.3)

    # Page 4: Sessions & Behavior
    story.append(PageBreak())
    story.append(Paragraph("Sessions & Behavior", styles["Heading1"]))
    story.append(Spacer(1, 0.08 * inch))

    sessions = compute_sessions(df, gap_minutes=SESSION_GAP_MINUTES)

    if sessions.empty:
        story.append(Paragraph("No sessions could be computed.", styles["Normal"]))
    else:
        duration_sessions = sessions[sessions["duration_minutes"] > 0].copy()

        if duration_sessions.empty:
            fig_duration = px.histogram(
                sessions.assign(duration_minutes=0),
                x="duration_minutes",
                nbins=20,
                title="Session Duration Distribution",
                labels={"duration_minutes": "Session Duration (Minutes)"},
            )
        else:
            fig_duration = px.histogram(
                duration_sessions,
                x="duration_minutes",
                nbins=40,
                title="Session Duration Distribution",
                labels={"duration_minutes": "Session Duration (Minutes)"},
            )

        fig_duration.update_layout(showlegend=False)

        fig_minutes = px.histogram(
            sessions,
            x="minutes",
            nbins=40,
            title="Minutes per Session Distribution",
            labels={"minutes": "Minutes per Session"},
        )
        fig_minutes.update_layout(showlegend=False)

        add_chart_pair(story, fig_duration, fig_minutes, width_inches=4.9, height_inches=2.6)

        fig_scatter = px.scatter(
            sessions,
            x="plays",
            y="minutes",
            title="Plays vs. Minutes per Session",
            labels={"plays": "Track Plays", "minutes": "Minutes"},
        )
        fig_scatter.update_layout(showlegend=False)

        add_single_chart(story, fig_scatter, width_inches=10.0, height_inches=2.2)

        longest = sessions.sort_values("minutes", ascending=False).head(8).copy()
        longest["session_date"] = longest["session_start"].dt.date

        longest_rows = [["Session Date", "Minutes", "Duration Minutes", "Track Plays"]]

        for _, row in longest.iterrows():
            longest_rows.append(
                [
                    clean_text(row["session_date"]),
                    f"{row['minutes']:,.1f}",
                    f"{row['duration_minutes']:,.1f}",
                    f"{int(row['plays']):,}",
                ]
            )

        add_table(
            story,
            longest_rows,
            col_widths=[2.0 * inch, 1.5 * inch, 2.0 * inch, 1.5 * inch],
        )

    doc.build(story)
    buffer.seek(0)

    return buffer.read()


