from io import BytesIO
from typing import Optional, Sequence

import pandas as pd
import plotly.express as px

from spotify_app.config import MIN_STREAM_SECONDS, SESSION_GAP_MINUTES
from spotify_app.data_transform import (
    compute_sessions,
    format_hour_label,
    period_settings,
    safe_group_count,
    safe_group_sum,
)


SPOTIFY_GREEN = "#1DB954"
SPOTIFY_LIGHT_GREEN = "#1ED760"
SPOTIFY_BLACK = "#191414"
SOFT_BG = "#0E1117"
CARD_BG = "#171B22"
TEXT_MUTED = "#A7B0BE"
GRID = "#2A313D"
AXIS_TEXT = "#E8EDF4"
TABLE_HEADER = "#2E77D0"

COLORWAY = [
    "#1DB954",
    "#2E77D0",
    "#9B5DE5",
    "#F15BB5",
    "#FF6B6B",
    "#FFA600",
    "#00BBF9",
    "#00C2A8",
    "#7CB342",
    "#5E60CE",
    "#F72585",
    "#4CC9F0",
]


def _clean_text(value, max_len: int = 48) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _safe_date(value) -> str:
    try:
        return pd.to_datetime(value).strftime("%b %d, %Y")
    except Exception:
        return str(value)


def _chart_topn(topn: int) -> int:
    """Keep exported charts readable while still respecting the selected Top N in normal use."""
    return max(5, min(int(topn), 20))


def _format_number_axes(fig):
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def _polish_export_fig(fig, *, show_legend: Optional[bool] = None):
    """Apply the shared visual language used by all exported Plotly charts."""
    existing_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    margin = {"l": 64, "r": 24, "t": 78, "b": 50}
    margin.update({key: value for key, value in existing_margin.items() if value is not None})
    margin["t"] = max(78, margin.get("t", 78))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        colorway=COLORWAY,
        font=dict(family="Arial", size=11, color=AXIS_TEXT),
        title=dict(font=dict(size=14, color=AXIS_TEXT), x=0.02, xanchor="left"),
        margin=margin,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="left",
            x=0,
            font=dict(size=9),
            title_text="",
        ),
        bargap=0.22,
    )

    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=11, color=AXIS_TEXT),
        tickfont=dict(size=10, color=AXIS_TEXT),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_font=dict(size=11, color=AXIS_TEXT),
        tickfont=dict(size=10, color=AXIS_TEXT),
        automargin=True,
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return _format_number_axes(fig)


def _label_bar_values(fig, axis: str = "x", decimals: int = 1):
    value_token = "x" if axis == "x" else "y"
    format_suffix = ":,.0f" if decimals == 0 else f":,.{decimals}f"
    fig.update_traces(
        texttemplate=f"%{{{value_token}{format_suffix}}}",
        textposition="outside",
        textfont=dict(size=9, color=AXIS_TEXT),
        cliponaxis=False,
        selector=dict(type="bar"),
    )
    return fig


def _fig_to_image(fig, width_inches: float, height_inches: float):
    from reportlab.lib.units import inch
    from reportlab.platypus import Image

    _polish_export_fig(fig)

    img_bytes = fig.to_image(
        format="png",
        width=int(width_inches * 180),
        height=int(height_inches * 180),
        scale=2,
    )

    return Image(BytesIO(img_bytes), width=width_inches * inch, height=height_inches * inch)


def _add_table(story, rows, col_widths=None, font_size: int = 8):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Spacer, Table, TableStyle

    if not rows:
        return

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEADER)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(AXIS_TEXT)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(CARD_BG), colors.HexColor("#1E2430")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))


def _add_chart_pair(story, fig_left, fig_right, width_inches=4.95, height_inches=2.8):
    from reportlab.lib.units import inch
    from reportlab.platypus import Spacer, Table, TableStyle

    left_img = _fig_to_image(fig_left, width_inches, height_inches)
    right_img = _fig_to_image(fig_right, width_inches, height_inches)

    chart_table = Table(
        [[left_img, right_img]],
        colWidths=[width_inches * inch, width_inches * inch],
    )
    chart_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(chart_table)
    story.append(Spacer(1, 0.12 * inch))


def _add_single_chart(story, fig, width_inches=10.1, height_inches=2.65):
    from reportlab.lib.units import inch
    from reportlab.platypus import Spacer

    story.append(_fig_to_image(fig, width_inches, height_inches))
    story.append(Spacer(1, 0.12 * inch))


def _section_header(story, title: str, subtitle: Optional[str], styles):
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer

    story.append(Paragraph(title, styles["SectionTitle"]))
    if subtitle:
        story.append(Paragraph(subtitle, styles["Muted"]))
    story.append(Spacer(1, 0.10 * inch))


def _metric_grid(story, metrics: Sequence[tuple[str, str]], styles, columns: int = 3):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    cells = []
    row = []
    for label, value in metrics:
        cell = [
            Paragraph(str(label), styles["MetricLabel"]),
            Paragraph(str(value), styles["MetricValue"]),
        ]
        row.append(cell)
        if len(row) == columns:
            cells.append(row)
            row = []

    if row:
        while len(row) < columns:
            row.append("")
        cells.append(row)

    table = Table(cells, colWidths=[(10.1 / columns) * inch] * columns)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CARD_BG)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.16 * inch))


def _page_background(canvas, doc):
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    canvas.saveState()
    canvas.setFillColor(colors.HexColor(SOFT_BG))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)

    canvas.setFillColor(colors.HexColor("#07090D"))
    canvas.rect(0, doc.pagesize[1] - 0.28 * inch, doc.pagesize[0], 0.28 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#2E77D0"))
    canvas.rect(0, doc.pagesize[1] - 0.32 * inch, doc.pagesize[0], 0.04 * inch, fill=1, stroke=0)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor(TEXT_MUTED))
    canvas.drawRightString(doc.pagesize[0] - 0.42 * inch, 0.24 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_shareable_pdf(
    df: pd.DataFrame,
    topn: int,
    selected_timezone_label: str = "Selected timezone",
    selected_year: Optional[str] = None,
    selected_day: Optional[str] = None,
    trend_granularity: str = "Week",
    artist_granularity: str = "Month",
) -> bytes:
    """Create one polished share snapshot PDF from the current filtered dashboard data."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    export_topn = _chart_topn(topn)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.42 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="HeroTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeroSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=22,
            textColor=colors.HexColor(AXIS_TEXT),
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor(TEXT_MUTED),
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(TEXT_MUTED),
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor(AXIS_TEXT),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(TEXT_MUTED),
        )
    )

    story = []

    date_source = "played_at_local" if "played_at_local" in df.columns else "played_at"
    start_date = _safe_date(pd.to_datetime(df[date_source]).min())
    end_date = _safe_date(pd.to_datetime(df[date_source]).max())
    filter_label = selected_year or "All years (Select all)"

    total_minutes = float(df["minutes"].sum())
    total_hours = total_minutes / 60.0
    track_plays = len(df)
    unique_artists = df["artist"].nunique()
    unique_tracks = df["track"].nunique()
    unique_albums = df["album"].nunique()

    top_artist = safe_group_sum(df, ["artist"], "minutes", 1)
    top_track = safe_group_sum(df, ["track", "artist"], "minutes", 1)
    top_album = safe_group_sum(df, ["album", "artist"], "minutes", 1)

    top_artist_name = _clean_text(top_artist.iloc[0]["artist"], 45) if not top_artist.empty else "N/A"
    top_track_name = (
        f"{_clean_text(top_track.iloc[0]['track'], 34)} — {_clean_text(top_track.iloc[0]['artist'], 24)}"
        if not top_track.empty
        else "N/A"
    )
    top_album_name = (
        f"{_clean_text(top_album.iloc[0]['album'], 34)} — {_clean_text(top_album.iloc[0]['artist'], 24)}"
        if not top_album.empty
        else "N/A"
    )

    by_hour_for_peak = df.groupby(["hour", "hour_label"], as_index=False)["minutes"].sum().sort_values("minutes", ascending=False)
    peak_hour = by_hour_for_peak.iloc[0]["hour_label"] if not by_hour_for_peak.empty else "N/A"

    by_dow_for_peak = df.groupby("day_of_week", as_index=False)["minutes"].sum().sort_values("minutes", ascending=False)
    peak_day = by_dow_for_peak.iloc[0]["day_of_week"] if not by_dow_for_peak.empty else "N/A"

    sessions = compute_sessions(df, gap_minutes=SESSION_GAP_MINUTES)
    longest_session = "N/A" if sessions.empty else f"{sessions['minutes'].max() / 60.0:,.2f} hr"

    # Page 1: Overview snapshot
    hero = Table(
        [
            [Paragraph("Spotify Statistics Share Snapshot", styles["HeroTitle"])],
            [Paragraph("Generated from the current filtered dashboard view", styles["HeroSub"])],
            [Paragraph(f"Selected date range: {start_date} to {end_date}", styles["HeroSub"])],
        ],
        colWidths=[10.1 * inch],
        rowHeights=[0.42 * inch, 0.25 * inch, 0.25 * inch],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#10151F")),
                ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#10151F")),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 0.18 * inch))

    metrics = [
        ("Total Hours", f"{total_hours:,.1f}"),
        ("Total Minutes", f"{total_minutes:,.0f}"),
        ("Track Plays", f"{track_plays:,}"),
        ("Unique Artists", f"{unique_artists:,}"),
        ("Unique Tracks", f"{unique_tracks:,}"),
        ("Unique Albums", f"{unique_albums:,}"),
        ("Top Artist", top_artist_name),
        ("Top Track", top_track_name),
        ("Top Album", top_album_name),
        ("Most Active Day", str(peak_day)),
        ("Most Active Hour", str(peak_hour)),
        ("Longest Session", longest_session),
    ]
    _metric_grid(story, metrics, styles, columns=3)

    story.append(
        Paragraph(
            "This snapshot is designed to be shared as one file. It uses the same filtered dataset as the dashboard, "
            "so the rankings, time patterns, trends, and session metrics reflect the current user selections.",
            styles["Muted"],
        )
    )

    # Page 2: Rankings
    story.append(PageBreak())
    _section_header(
        story,
        "Rankings",
        "Top artists, tracks, albums, and play-count rankings for the current filtered view.",
        styles,
    )

    top_artists = safe_group_sum(df, ["artist"], "minutes", export_topn).copy()
    top_artists["hours"] = top_artists["minutes"] / 60.0
    fig_artists = px.bar(
        top_artists.sort_values("hours", ascending=True),
        x="hours",
        y="artist",
        orientation="h",
        title="Top Artists by Hours",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"hours": "Hours", "artist": "Artist"},
    )
    _label_bar_values(fig_artists, axis="x", decimals=1)
    fig_artists.update_layout(showlegend=False, margin=dict(l=170, r=20, t=58, b=45))

    top_artists_plays = safe_group_count(df, ["artist"], export_topn, name="plays").copy()
    fig_artist_plays = px.bar(
        top_artists_plays.sort_values("plays", ascending=True),
        x="plays",
        y="artist",
        orientation="h",
        title="Top Artists by Plays",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"artist": "Artist", "plays": "Plays"},
    )
    _label_bar_values(fig_artist_plays, axis="x", decimals=0)
    fig_artist_plays.update_layout(showlegend=False, margin=dict(l=170, r=20, t=58, b=45))

    _add_chart_pair(story, fig_artists, fig_artist_plays, height_inches=3.05)

    top_tracks = safe_group_sum(df, ["track", "artist", "album"], "minutes", export_topn).copy()
    top_tracks["hours"] = top_tracks["minutes"] / 60.0
    top_tracks["track_label"] = top_tracks.apply(
        lambda r: f"{_clean_text(r['track'], 25)} — {_clean_text(r['artist'], 18)}",
        axis=1,
    )
    fig_tracks = px.bar(
        top_tracks.sort_values("hours", ascending=True),
        x="hours",
        y="track_label",
        orientation="h",
        title="Top Tracks by Hours",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"hours": "Hours", "track_label": "Track"},
    )
    _label_bar_values(fig_tracks, axis="x", decimals=1)
    fig_tracks.update_layout(showlegend=False, margin=dict(l=205, r=20, t=58, b=45))

    top_tracks_plays = safe_group_count(df, ["track", "artist", "album"], export_topn, name="plays").copy()
    top_tracks_plays["track_label"] = top_tracks_plays.apply(
        lambda r: f"{_clean_text(r['track'], 25)} - {_clean_text(r['artist'], 18)}",
        axis=1,
    )
    fig_tracks_plays = px.bar(
        top_tracks_plays.sort_values("plays", ascending=True),
        x="plays",
        y="track_label",
        orientation="h",
        title="Top Tracks by Plays",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"plays": "Plays", "track_label": "Track"},
    )
    _label_bar_values(fig_tracks_plays, axis="x", decimals=0)
    fig_tracks_plays.update_layout(showlegend=False, margin=dict(l=205, r=20, t=58, b=45))

    _add_chart_pair(story, fig_tracks, fig_tracks_plays, height_inches=3.05)
    story.append(PageBreak())
    _section_header(
        story,
        "Album Rankings",
        "Album listening ranked by total time and play count.",
        styles,
    )

    top_albums = safe_group_sum(df, ["album", "artist"], "minutes", export_topn).copy()
    top_albums["hours"] = top_albums["minutes"] / 60.0
    top_albums["album_label"] = top_albums.apply(
        lambda r: f"{_clean_text(r['album'], 26)} — {_clean_text(r['artist'], 18)}",
        axis=1,
    )
    fig_albums = px.bar(
        top_albums.sort_values("hours", ascending=True),
        x="hours",
        y="album_label",
        orientation="h",
        title="Top Albums by Hours",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"hours": "Hours", "album_label": "Album"},
    )
    _label_bar_values(fig_albums, axis="x", decimals=1)
    fig_albums.update_layout(showlegend=False, margin=dict(l=205, r=20, t=58, b=45))

    top_albums_plays = safe_group_count(df, ["album", "artist"], export_topn, name="plays").copy()
    top_albums_plays["album_label"] = top_albums_plays.apply(
        lambda r: f"{_clean_text(r['album'], 26)} - {_clean_text(r['artist'], 18)}",
        axis=1,
    )
    fig_albums_plays = px.bar(
        top_albums_plays.sort_values("plays", ascending=True),
        x="plays",
        y="album_label",
        orientation="h",
        title="Top Albums by Plays",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"album_label": "Album", "plays": "Plays"},
    )
    _label_bar_values(fig_albums_plays, axis="x", decimals=0)
    fig_albums_plays.update_layout(showlegend=False, margin=dict(l=205, r=20, t=58, b=45))

    _add_chart_pair(story, fig_albums, fig_albums_plays, height_inches=3.05)
    story.append(PageBreak())

    # Page 3: Behavior details
    _section_header(
        story,
        "Repeat Behavior & Peak Days",
        "Repeat distribution, artist peak days, most-listened days, and repeated-song streaks.",
        styles,
    )

    plays_per_track = (
        df.groupby(["track", "artist"], as_index=False)
        .size()
        .rename(columns={"size": "plays"})
    )
    repeat_dist = (
        plays_per_track.groupby("plays", as_index=False)
        .size()
        .rename(columns={"size": "number_of_tracks"})
        .sort_values("plays")
    )
    fig_repeat_dist = px.bar(
        repeat_dist,
        x="plays",
        y="number_of_tracks",
        title="Track Repeat Distribution",
        color_discrete_sequence=[SPOTIFY_GREEN],
        labels={"plays": "Plays per Track", "number_of_tracks": "Number of Tracks"},
    )
    fig_repeat_dist.update_traces(marker_color=SPOTIFY_GREEN)

    daily_artist = df.groupby(["artist", "day"], as_index=False)["minutes"].sum()
    idx = daily_artist.groupby("artist")["minutes"].idxmax()
    artist_peaks = daily_artist.loc[idx].copy()
    artist_peaks = artist_peaks.sort_values("minutes", ascending=False).head(export_topn).copy()
    artist_peaks["date"] = artist_peaks["day"].dt.date
    artist_peaks["label"] = artist_peaks["artist"] + " — " + artist_peaks["date"].astype(str)
    artist_peaks["hours"] = artist_peaks["minutes"] / 60.0

    fig_artist_peaks = px.bar(
        artist_peaks.sort_values("hours", ascending=True),
        x="hours",
        y="label",
        orientation="h",
        title="Artist Peak Days",
        color="artist",
        color_discrete_sequence=COLORWAY,
        labels={"hours": "Hours", "label": "Artist and Date"},
    )
    _label_bar_values(fig_artist_peaks, axis="x", decimals=1)
    fig_artist_peaks.update_layout(showlegend=False, margin=dict(l=225, r=20, t=58, b=45))

    _add_chart_pair(story, fig_repeat_dist, fig_artist_peaks, height_inches=2.85)

    daily_total = (
        df.groupby("day", as_index=False)
        .agg(
            total_minutes=("minutes", "sum"),
            number_of_artists=("artist", "nunique"),
            plays=("track", "size"),
        )
        .sort_values("total_minutes", ascending=False)
        .head(12)
        .copy()
    )
    daily_total["date"] = daily_total["day"].dt.date
    daily_rows = [["Date", "Total Hours", "Artists", "Track Plays"]]
    for _, row in daily_total.iterrows():
        daily_rows.append(
            [
                str(row["date"]),
                f"{row['total_minutes'] / 60.0:,.2f}",
                f"{int(row['number_of_artists']):,}",
                f"{int(row['plays']):,}",
            ]
        )

    repeat_sequence = df.sort_values("played_at").copy()
    repeat_sequence["track_key"] = repeat_sequence["track"].astype(str) + " — " + repeat_sequence["artist"].astype(str)
    repeat_sequence["new_streak"] = repeat_sequence["track_key"] != repeat_sequence["track_key"].shift()
    repeat_sequence["streak_id"] = repeat_sequence["new_streak"].cumsum()
    streaks = (
        repeat_sequence.groupby("streak_id", as_index=False)
        .agg(
            track=("track", "first"),
            artist=("artist", "first"),
            repeat_count=("track", "size"),
            streak_start=("played_at_local", "min"),
            streak_end=("played_at_local", "max"),
        )
    )
    streaks = streaks[streaks["repeat_count"] > 1].copy()
    streaks = streaks.sort_values("repeat_count", ascending=False).head(12)

    repeated_rows = [["Track", "Artist", "Consecutive Plays", "Start"]]
    if streaks.empty:
        repeated_rows.append(["No consecutive repeated tracks found", "", "", ""])
    else:
        for _, row in streaks.iterrows():
            repeated_rows.append(
                [
                    _clean_text(row["track"], 28),
                    _clean_text(row["artist"], 20),
                    f"{int(row['repeat_count']):,}",
                    pd.to_datetime(row["streak_start"]).strftime("%b %d, %Y"),
                ]
            )

    tables = Table(
        [
            [Paragraph("Most-listened days", styles["SectionTitle"]), Paragraph("Repeated songs", styles["SectionTitle"])],
            [
                Table(daily_rows, colWidths=[1.15 * inch, 1.10 * inch, 0.75 * inch, 0.85 * inch]),
                Table(repeated_rows, colWidths=[1.65 * inch, 1.15 * inch, 0.90 * inch, 1.05 * inch]),
            ],
        ],
        colWidths=[4.2 * inch, 5.75 * inch],
    )
    tables.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(tables)

    # Style nested tables after creation
    for nested in [tables._cellvalues[1][0], tables._cellvalues[1][1]]:
        nested.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEADER)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(AXIS_TEXT)),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(CARD_BG), colors.HexColor("#1E2430")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )

    # Page 4: Time patterns
    story.append(PageBreak())
    _section_header(
        story,
        "Time Patterns",
        f"Day-of-week and hour-of-day listening patterns using {selected_timezone_label}.",
        styles,
    )

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour_order = [format_hour_label(h) for h in range(24)]

    by_dow = df.groupby("day_of_week", as_index=False)["minutes"].sum()
    by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=order, ordered=True)
    by_dow = by_dow.sort_values("day_of_week")
    by_dow["hours"] = by_dow["minutes"] / 60.0

    fig_dow = px.bar(
        by_dow,
        x="day_of_week",
        y="hours",
        title="Hours by Day of Week",
        color="day_of_week",
        color_discrete_sequence=COLORWAY,
        labels={"day_of_week": "Day", "hours": "Hours"},
    )
    _label_bar_values(fig_dow, axis="y", decimals=1)
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
        color="hour",
        color_continuous_scale="Greens",
        labels={"hour_label": "Hour", "hours": "Hours"},
    )
    _label_bar_values(fig_hour, axis="y", decimals=1)
    fig_hour.update_layout(showlegend=False, coloraxis_showscale=False)

    _add_chart_pair(story, fig_dow, fig_hour, height_inches=3.0)

    heatmap_data = (
        df.groupby(["day_of_week", "hour"], as_index=False)["minutes"].sum()
        .pivot(index="day_of_week", columns="hour", values="minutes")
        .reindex(order)
        .reindex(columns=list(range(24)))
        .fillna(0)
    )
    heatmap_hours = heatmap_data / 60.0

    fig_heatmap = px.imshow(
        heatmap_hours,
        aspect="auto",
        title="Listening Heatmap by Day and Hour",
        labels=dict(x="Hour of Day", y="Day of Week", color="Hours"),
        x=hour_order,
        y=order,
        color_continuous_scale="Greens",
    )
    fig_heatmap.update_layout(
        margin=dict(l=92, r=46, t=58, b=40),
        coloraxis_colorbar=dict(title="Hours", tickfont=dict(size=9)),
        xaxis=dict(side="top", tickangle=0, showgrid=False),
        yaxis=dict(showgrid=False),
    )
    fig_heatmap.update_xaxes(tickfont=dict(size=8))
    fig_heatmap.update_yaxes(tickfont=dict(size=10))

    _add_single_chart(story, fig_heatmap, height_inches=2.35)

    # Detail for selected or busiest day
    if selected_day is not None:
        detail_date = pd.to_datetime(selected_day).date()
    elif not daily_total.empty:
        detail_date = daily_total.iloc[0]["date"]
    else:
        detail_date = None

    if detail_date is not None:
        day_detail = df[df["date"] == detail_date].copy()
        day_artists = (
            day_detail.groupby("artist", as_index=False)["minutes"].sum()
            .sort_values("minutes", ascending=False)
            .head(12)
        )
        day_artists["hours"] = day_artists["minutes"] / 60.0
        fig_day_detail = px.bar(
            day_artists.sort_values("hours", ascending=True),
            x="hours",
            y="artist",
            orientation="h",
            title=f"Top Artists on {detail_date}",
            color="artist",
            color_discrete_sequence=COLORWAY,
            labels={"artist": "Artist", "hours": "Hours"},
        )
        _label_bar_values(fig_day_detail, axis="x", decimals=1)
        fig_day_detail.update_layout(showlegend=False, margin=dict(l=180, r=20, t=58, b=45))

        day_tracks = (
            day_detail.groupby(["track", "artist"], as_index=False)
            .size()
            .rename(columns={"size": "track_plays"})
            .sort_values("track_plays", ascending=False)
            .head(10)
        )
        track_rows = [["Track", "Artist", "Plays"]]
        for _, row in day_tracks.iterrows():
            track_rows.append([
                _clean_text(row["track"], 32),
                _clean_text(row["artist"], 22),
                f"{int(row['track_plays']):,}",
            ])

        chart = _fig_to_image(fig_day_detail, 5.0, 3.0)
        tracks_table = Table(track_rows, colWidths=[2.15 * inch, 1.55 * inch, 0.55 * inch])
        tracks_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEADER)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(AXIS_TEXT)),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(CARD_BG), colors.HexColor("#1E2430")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(
            Table(
                [[chart, tracks_table]],
                colWidths=[5.1 * inch, 4.8 * inch],
                style=[("VALIGN", (0, 0), (-1, -1), "TOP")],
            )
        )

    # Page 5: Trends
    story.append(PageBreak())
    _section_header(
        story,
        "Trends",
        "Listening trend, cumulative listening, artist trends, artist diversity, and new artist discovery.",
        styles,
    )

    period_col, period_label = period_settings(trend_granularity)
    listening_trend = df.groupby(period_col, as_index=False)["minutes"].sum().sort_values(period_col)
    listening_trend["hours"] = listening_trend["minutes"] / 60.0

    fig_listening = px.line(
        listening_trend,
        x=period_col,
        y="hours",
        title=f"Hours Over Time by {period_label}",
        labels={period_col: period_label, "hours": "Hours"},
    )
    fig_listening.update_traces(line=dict(color=SPOTIFY_GREEN, width=3), mode="lines+markers")
    fig_listening.update_yaxes(tickformat=",.1f")

    cumulative = listening_trend.copy()
    cumulative["cumulative_hours"] = cumulative["hours"].cumsum()
    fig_cumulative = px.line(
        cumulative,
        x=period_col,
        y="cumulative_hours",
        title=f"Cumulative Hours by {period_label}",
        labels={period_col: period_label, "cumulative_hours": "Cumulative Hours"},
    )
    fig_cumulative.update_traces(line=dict(color="#2E77D0", width=3), mode="lines+markers")
    fig_cumulative.update_yaxes(tickformat=",.1f")

    _add_chart_pair(story, fig_listening, fig_cumulative, height_inches=2.75)

    artist_period_col, artist_period_label = period_settings(artist_granularity)
    top_artist_list = (
        df.groupby("artist", as_index=False)["minutes"].sum()
        .sort_values("minutes", ascending=False)
        .head(8)["artist"]
        .tolist()
    )
    trend = (
        df[df["artist"].isin(top_artist_list)]
        .groupby([artist_period_col, "artist"], as_index=False)["minutes"].sum()
        .sort_values(artist_period_col)
    )
    trend["hours"] = trend["minutes"] / 60.0

    fig_artist_trend = px.area(
        trend,
        x=artist_period_col,
        y="hours",
        color="artist",
        title=f"Top Artists Over Time by {artist_period_label}",
        color_discrete_sequence=COLORWAY,
        labels={artist_period_col: artist_period_label, "hours": "Hours", "artist": "Artist"},
    )

    diversity = (
        df.groupby(artist_period_col, as_index=False)
        .agg(unique_artists=("artist", "nunique"))
        .sort_values(artist_period_col)
    )
    fig_diversity = px.line(
        diversity,
        x=artist_period_col,
        y="unique_artists",
        title=f"Artist Diversity by {artist_period_label}",
        labels={artist_period_col: artist_period_label, "unique_artists": "Unique Artists"},
    )
    fig_diversity.update_traces(line=dict(color="#9B5DE5", width=3), mode="lines+markers")

    _add_chart_pair(story, fig_artist_trend, fig_diversity, height_inches=2.75)

    first_artist = (
        df.sort_values("played_at")
        .groupby("artist", as_index=False)
        .first()[["artist", "played_at"]]
    )
    if artist_granularity == "Month":
        first_artist["period"] = first_artist["played_at"].dt.strftime("%Y-%m")
        discovery_period_label = "Month"
    else:
        first_artist["period"] = first_artist["played_at"].dt.year
        discovery_period_label = "Year"

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
        title=f"New Artists Discovered by {discovery_period_label}",
        color="period",
        color_discrete_sequence=COLORWAY,
        labels={"period": discovery_period_label, "new_artists": "New Artists"},
    )
    _label_bar_values(fig_discovery, axis="y", decimals=0)
    fig_discovery.update_layout(showlegend=False)
    _add_single_chart(story, fig_discovery, height_inches=2.2)

    # Page 6: Sessions & Behavior
    story.append(PageBreak())
    _section_header(
        story,
        "Sessions & Behavior",
        f"Listening sessions are inferred using a {SESSION_GAP_MINUTES}-minute inactivity gap.",
        styles,
    )

    if sessions.empty:
        story.append(Paragraph("No sessions could be computed for this selection.", styles["Muted"]))
    else:
        duration_sessions = sessions[sessions["duration_minutes"] > 0].copy()
        if duration_sessions.empty:
            duration_sessions = sessions.assign(duration_hours=0)
            duration_title = "Session Duration Distribution"
        else:
            duration_sessions["duration_hours"] = duration_sessions["duration_minutes"] / 60.0
            duration_title = "Multi-track Session Duration Distribution"

        fig_duration = px.histogram(
            duration_sessions,
            x="duration_hours",
            nbins=40,
            title=duration_title,
            labels={"duration_hours": "Session Span in Hours"},
        )
        fig_duration.update_traces(marker_color=SPOTIFY_GREEN)

        sessions_hours = sessions.copy()
        sessions_hours["hours"] = sessions_hours["minutes"] / 60.0

        fig_minutes = px.histogram(
            sessions_hours,
            x="hours",
            nbins=40,
            title="Hours per Session Distribution",
            labels={"hours": "Hours per Session"},
        )
        fig_minutes.update_traces(marker_color="#2E77D0")

        _add_chart_pair(story, fig_duration, fig_minutes, height_inches=2.65)

        fig_scatter = px.scatter(
            sessions_hours,
            x="plays",
            y="hours",
            title="Plays vs. Hours per Session",
            color="hours",
            color_continuous_scale="Greens",
            labels={"plays": "Track Plays", "hours": "Hours"},
        )
        fig_scatter.update_layout(coloraxis_showscale=False)

        longest = sessions.sort_values("minutes", ascending=False).head(12).copy()
        longest["session_date"] = longest["session_start"].dt.date
        longest_rows = [["Session Date", "Hours", "Track Plays"]]
        for _, row in longest.iterrows():
            longest_rows.append(
                [
                    str(row["session_date"]),
                    f"{row['minutes'] / 60.0:,.2f}",
                    f"{int(row['plays']):,}",
                ]
            )

        scatter_img = _fig_to_image(fig_scatter, 5.1, 3.0)
        longest_table = Table(longest_rows, colWidths=[1.45 * inch, 1.05 * inch, 1.10 * inch])
        longest_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEADER)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(AXIS_TEXT)),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(CARD_BG), colors.HexColor("#1E2430")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(
            Table(
                [[scatter_img, longest_table]],
                colWidths=[5.25 * inch, 4.7 * inch],
                style=[("VALIGN", (0, 0), (-1, -1), "TOP")],
            )
        )

    # Page 7: About snapshot
    story.append(PageBreak())
    _section_header(
        story,
        "About This Snapshot",
        "A compact explanation of the dataset, fields, filters, and privacy behavior.",
        styles,
    )

    about_rows = [
        ["Topic", "Details"],
        ["Timezone", selected_timezone_label],
        ["Year filter", str(filter_label)],
        ["Top N selected", str(topn)],
        ["Top chart items exported", str(export_topn)],
        ["Stream filter", f">= {MIN_STREAM_SECONDS} seconds"],
        ["Session gap", f"> {SESSION_GAP_MINUTES} minutes"],
        ["Source data", "Spotify Extended Streaming History export ZIP"],
        ["Fields used", "ts, ms_played, master_metadata_track_name, master_metadata_album_artist_name, master_metadata_album_album_name"],
        ["Renamed fields", "played_at, ms_played, track, artist, album"],
        ["Derived fields", "minutes, date, day, week, month, year, hour, day_of_week, played_at_local"],
        ["Time grouping", f"Date, day-of-week, hour, and most-listened-day groupings use {selected_timezone_label}."],
        ["Sessions", f"A new session starts after more than {SESSION_GAP_MINUTES} minutes of inactivity."],
        ["Privacy", "The app does not require Spotify login and uses uploaded data only for the active Streamlit session."],
    ]
    _add_table(story, about_rows, col_widths=[1.7 * inch, 8.3 * inch], font_size=8)

    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "Created with Spotify Statistics by Jonah Jutzi. This PDF is a share snapshot of the current dashboard view, "
            "not a full copy of the raw Spotify export.",
            styles["Muted"],
        )
    )

    doc.build(story, onFirstPage=_page_background, onLaterPages=_page_background)
    buffer.seek(0)
    return buffer.read()
