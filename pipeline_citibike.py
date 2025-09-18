"""
Pipeline for Citi Bike: transform -> views -> validate -> export (BI)

Run (from repo root, after putting CSVs in data/raw/citibike/):
  python src/ingest_citibike.py
  python run_pipeline_citibike.py

Outputs (in ./outputs):
  - monthly_kpis.csv       (rides, avg/median duration, avg distance, member vs casual)
  - top_stations.csv       (top 50 start stations)
  - hourly_profile.csv     (rides by hour 0..23)
  - duration_hist.csv      (5-minute bins, 0..120)
  - distance_hist.csv      (1-mile bins, 0..20; only if distance available)
  - stations.csv           (station id/name to lat/lon, if coordinates exist)
"""

from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np

DB = Path("data/processed/citibike.sqlite")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Steps ----------
def transform(conn: sqlite3.Connection) -> None:
    """Read raw trips, derive fields, filter outliers, and persist 'trips_clean'."""
    df = pd.read_sql("SELECT * FROM trips", conn, parse_dates=["started_at", "ended_at"])
    if df.empty:
        raise SystemExit("No trips found. Did you run: python src/ingest_citibike.py ?")

    # Derive year_month and hour-of-day
    df["year_month"] = df["started_at"].dt.to_period("M").astype(str)
    df["hour"] = df["started_at"].dt.hour

    # Quality filters (sane ride ranges)
    # Keep 1–120 minutes; distance 0–20 miles when present
    df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 120)]
    if "distance_miles" in df.columns:
        df = df[df["distance_miles"].isna() | ((df["distance_miles"] >= 0) & (df["distance_miles"] <= 20))]

    # Persist cleaned data
    df.to_sql("trips_clean", conn, if_exists="replace", index=False)


def create_views(conn: sqlite3.Connection) -> None:
    """Create handy SQL views (fast to query, easy to inspect)."""
    cur = conn.cursor()
    cur.executescript(
        """
        -- Monthly base metrics (no median here; we'll compute medians in Python)
        DROP VIEW IF EXISTS v_monthly_base;
        CREATE VIEW v_monthly_base AS
        SELECT
          year_month,
          COUNT(*) AS rides,
          AVG(duration_min) AS avg_duration_min,
          AVG(distance_miles) AS avg_distance_miles,
          SUM(CASE WHEN member_type='member' THEN 1 ELSE 0 END) AS rides_member,
          SUM(CASE WHEN member_type='casual' THEN 1 ELSE 0 END) AS rides_casual
        FROM trips_clean
        GROUP BY year_month;

        -- Top 50 start stations by ride count
        DROP VIEW IF EXISTS v_top_stations;
        CREATE VIEW v_top_stations AS
        SELECT
          start_station_id,
          start_station_name,
          COUNT(*) AS rides
        FROM trips_clean
        WHERE start_station_id IS NOT NULL
        GROUP BY start_station_id, start_station_name
        ORDER BY rides DESC
        LIMIT 50;

        -- Hourly ride profile (0..23)
        DROP VIEW IF EXISTS v_hourly_profile;
        CREATE VIEW v_hourly_profile AS
        SELECT hour, COUNT(*) AS rides
        FROM trips_clean
        GROUP BY hour
        ORDER BY hour;
        """
    )
    conn.commit()


def validate(conn: sqlite3.Connection) -> None:
    """Basic sanity checks."""
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM trips_clean;").fetchone()[0]
    months = cur.execute("SELECT COUNT(DISTINCT year_month) FROM trips_clean;").fetchone()[0]
    print("VALIDATION:", {"rows": total, "distinct_months": months})
    assert total > 0, "No rows after transform."
    assert months > 0, "No month coverage (are started_at timestamps valid?)"


def export_for_bi(conn: sqlite3.Connection) -> None:
    """Export BI-ready CSVs, including medians computed in Python."""
    # Monthly KPIs with medians (computed in Python)
    trips = pd.read_sql(
        "SELECT year_month, duration_min, distance_miles, member_type, "
        "start_station_id, start_station_name, "
        "CAST(NULL AS REAL) AS start_lat, CAST(NULL AS REAL) AS start_lng "
        "FROM trips_clean",
        conn
    )
    # (Note: start_lat/lng may be present in your schema; if so, they'll come through.)

    monthly = (
        trips.groupby("year_month", as_index=False)
        .agg(
            rides=("duration_min", "count"),
            avg_duration_min=("duration_min", "mean"),
            median_duration_min=("duration_min", "median"),
            avg_distance_miles=("distance_miles", "mean"),
            rides_member=("member_type", lambda s: (s == "member").sum()),
            rides_casual=("member_type", lambda s: (s == "casual").sum()),
        )
        .sort_values("year_month")
    )
    monthly["avg_duration_min"] = monthly["avg_duration_min"].round(2)
    monthly["median_duration_min"] = monthly["median_duration_min"].round(2)
    monthly["avg_distance_miles"] = monthly["avg_distance_miles"].round(2)
    monthly.to_csv(OUT / "monthly_kpis.csv", index=False)

    # Top stations (from view)
    pd.read_sql("SELECT * FROM v_top_stations", conn).to_csv(OUT / "top_stations.csv", index=False)

    # Hourly profile (from view)
    pd.read_sql("SELECT * FROM v_hourly_profile", conn).to_csv(OUT / "hourly_profile.csv", index=False)

    # Histograms (pre-binned for easy visuals)
    # Duration 0..120 in 5-min bins
    dur = trips["duration_min"].dropna().clip(0, 120)
    dur_bins = np.arange(0, 125, 5)
    dur_counts, _ = np.histogram(dur, bins=dur_bins)
    duration_hist = pd.DataFrame({
        "duration_min_bin": pd.IntervalIndex.from_breaks(dur_bins).astype(str),
        "count": dur_counts
    })
    duration_hist.to_csv(OUT / "duration_hist.csv", index=False)

    # Distance 0..20 in 1-mile bins (if present)
    if "distance_miles" in trips.columns and trips["distance_miles"].notna().any():
        dist = trips["distance_miles"].dropna().clip(0, 20)
        dist_bins = np.arange(0, 21, 1)
        dist_counts, _ = np.histogram(dist, bins=dist_bins)
        distance_hist = pd.DataFrame({
            "distance_miles_bin": pd.IntervalIndex.from_breaks(dist_bins).astype(str),
            "count": dist_counts
        })
        distance_hist.to_csv(OUT / "distance_hist.csv", index=False)

    # --- Station coordinates export (for maps) ---
    # If your ingested data contains start_lat/start_lng, write a stations.csv
    # Try to load coords directly from trips_clean (handles months with coords)
    try:
        trips_full = pd.read_sql("SELECT start_station_id, start_station_name, start_lat, start_lng FROM trips_clean", conn)
        if {"start_lat", "start_lng"}.issubset(trips_full.columns):
            stations = (
                trips_full.dropna(subset=["start_station_id"])
                          .groupby(["start_station_id", "start_station_name"], as_index=False)
                          .agg(lat=("start_lat", "first"), lon=("start_lng", "first"))
            )
            # Basic sanity: drop rows with missing lat/lon
            stations = stations.dropna(subset=["lat", "lon"])
            if not stations.empty:
                stations.to_csv(OUT / "stations.csv", index=False)
                print(f"Exported stations.csv with {len(stations)} rows")
    except Exception:
        # Silently skip if columns not present in your schema
        pass

    print("Exported:", [p.name for p in OUT.glob("*.csv")])


def main():
    if not DB.exists():
        raise SystemExit("Missing DB. Run ingest first: python src/ingest_citibike.py")
    with sqlite3.connect(DB) as conn:
        transform(conn)
        create_views(conn)
        validate(conn)
        export_for_bi(conn)
    print("Pipeline finished:", datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
