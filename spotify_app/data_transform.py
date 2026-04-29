from typing import Optional

import pandas as pd


def auto_hbar_height(
    n_rows: int,
    min_h: int = 450,
    per_row: int = 26,
    pad: int = 140,
    max_h: int = 2200,
) -> int:
    """Make horizontal bar charts tall enough so labels do not get cut off when Top N grows."""
    return min(max_h, max(min_h, pad + per_row * max(1, n_rows)))


def format_hour_label(hour: int) -> str:
    """Convert 0-23 hour to user-friendly time labels."""
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def add_time_fields(df: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """Add reusable local time fields for charts."""
    d = df.copy()

    local_time = d["played_at"].dt.tz_convert(timezone).dt.tz_localize(None)

    d["played_at_local"] = local_time
    d["date"] = local_time.dt.date
    d["day"] = local_time.dt.floor("D")
    d["week"] = local_time.dt.to_period("W").dt.start_time
    d["month"] = local_time.dt.strftime("%Y-%m")
    d["year"] = local_time.dt.year.astype("int64")

    d["hour"] = local_time.dt.hour
    d["hour_label"] = d["hour"].apply(format_hour_label)

    d["day_of_week"] = local_time.dt.day_name()
    d["dow"] = d["day_of_week"]

    return d


def safe_group_sum(
    df: pd.DataFrame,
    group_cols,
    value_col: str,
    topn: Optional[int] = None,
) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, as_index=False)[value_col].sum()
        .sort_values(value_col, ascending=False)
    )

    return out.head(topn) if topn else out


def safe_group_count(
    df: pd.DataFrame,
    group_cols,
    topn: Optional[int] = None,
    name: str = "count",
) -> pd.DataFrame:
    out = df.groupby(group_cols, as_index=False).size().rename(columns={"size": name})
    out = out.sort_values(name, ascending=False)

    return out.head(topn) if topn else out


def compute_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Create listening sessions based on inactivity gaps."""
    if df["played_at"].isna().all():
        return pd.DataFrame()

    d = df.sort_values("played_at").copy()
    gaps = d["played_at"].diff().dt.total_seconds().fillna(0) / 60.0

    d["new_session"] = (gaps > gap_minutes).astype(int)
    d["session_id"] = d["new_session"].cumsum()

    sessions = d.groupby("session_id", as_index=False).agg(
        session_start=("played_at", "min"),
        session_end=("played_at", "max"),
        plays=("played_at", "size"),
        minutes=("minutes", "sum"),
    )

    sessions["duration_minutes"] = (
        sessions["session_end"] - sessions["session_start"]
    ).dt.total_seconds() / 60.0

    sessions["session_date"] = sessions["session_start"].dt.date

    return sessions.sort_values("session_start")


def period_settings(granularity: str):
    """Return the dataframe column and display label for a selected time grouping."""
    if granularity == "Day":
        return "day", "Day"

    if granularity == "Week":
        return "week", "Week"

    if granularity == "Month":
        return "month", "Month"

    return "year", "Year"
